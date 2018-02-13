#include <cuda.h>
#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_cmd.h>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include "ctrl.h"
#include "buffer.h"
#include "settings.h"
#include "event.h"
#include "queue.h"
#ifdef __DIS_CLUSTER__
#include <sisci_api.h>
#endif

using error = std::runtime_error;
using std::string;



__device__ static
void moveBytes(const void* src, size_t srcOffset, void* dst, size_t dstOffset, size_t size)
{
    const uint16_t numThreads = blockDim.x;
    const uint16_t threadNum = threadIdx.x;

    const ulong4* source = (ulong4*) (((const unsigned char*) src) + srcOffset);
    ulong4* destination = (ulong4*) (((unsigned char*) dst) + dstOffset);

    for (size_t i = 0, n = size / sizeof(ulong4); i < n; i += numThreads)
    {
        destination[i + threadNum] = source[i + threadNum];
    }
}


__device__ static
void waitForIoCompletion(nvm_queue_t* cq, nvm_queue_t* sq, uint64_t* errCount)
{
    const uint16_t numThreads = blockDim.x;

    for (uint16_t i = 0; i < numThreads; ++i)
    {
        nvm_cpl_t* cpl = nullptr;
        while ((cpl = nvm_cq_dequeue(cq)) == nullptr);

        nvm_sq_update(sq);

        if (!NVM_ERR_OK(cpl))
        {
            *errCount = *errCount + 1;
        }
    }

    nvm_cq_update(cq);
}


__device__ static
nvm_cmd_t* prepareChunk(QueuePair* qp, nvm_cmd_t* last, const uint64_t ioaddr, uint16_t offset, uint64_t blockOffset, uint32_t currChunk)
{
    const uint16_t numThreads = blockDim.x;
    const uint16_t threadNum = threadIdx.x;
    const uint16_t threadOffset = threadNum + numThreads * offset;

    const uint32_t pageSize = qp->pageSize;
    const uint32_t blockSize = qp->blockSize;
    const uint32_t nvmNamespace = qp->nvmNamespace;
    const uint32_t chunkPages = qp->pagesPerChunk;

    // Calculate offsets
    const uint16_t blocksPerChunk = NVM_PAGE_TO_BLOCK(pageSize, blockSize, chunkPages);
    const uint64_t currBlock = NVM_PAGE_TO_BLOCK(pageSize, blockSize, (currChunk + threadNum) * chunkPages);

    // Prepare PRP list building
    void* prpList = NVM_PTR_OFFSET(qp->prpList, pageSize, threadOffset);
    uint64_t prpListAddr = NVM_ADDR_OFFSET(qp->prpListIoAddr, pageSize, threadOffset);

    uint64_t addrs[0x1000 / sizeof(uint64_t)]; // FIXME: This assumes that page size is 4K
    for (uint32_t page = 0; page < chunkPages; ++page)
    {
        addrs[page] = NVM_ADDR_OFFSET(ioaddr, pageSize, chunkPages * threadOffset + page);
    }

    // Enqueue commands
    nvm_cmd_t* cmd = nvm_sq_enqueue_n(&qp->sq, last, numThreads, threadNum);

    // Set command fields
    nvm_cmd_header(cmd, NVM_IO_READ, nvmNamespace);
    nvm_cmd_data(cmd, pageSize, chunkPages, prpList, prpListAddr, addrs);
    nvm_cmd_rw_blks(cmd, currBlock + blockOffset, blocksPerChunk);

    __threadfence();
    return cmd;
}



__global__ static
void readDoubleBuffered(QueuePair* qp, const uint64_t ioaddr, void* src, void* dst, size_t numChunks, uint64_t* errCount)
{
    const uint16_t numThreads = blockDim.x;
    const uint16_t threadNum = threadIdx.x;
    const uint32_t pageSize = qp->pageSize;
    const size_t chunkSize = qp->pagesPerChunk * pageSize;
    nvm_queue_t* sq = &qp->sq;

    uint64_t blockOffset = 0; // TODO: Fix this

    uint32_t currChunk = 0;
    bool bufferOffset = false;

    nvm_cmd_t* last = prepareChunk(qp, nullptr, ioaddr, bufferOffset, blockOffset, currChunk);

    if (threadNum == 0)
    {
        *errCount = 0;
        nvm_sq_submit(sq);
    }
    __syncthreads();

    while (currChunk + numThreads < numChunks)
    {
        // Prepare in advance next chunk
        last = prepareChunk(qp, last, ioaddr, !bufferOffset, blockOffset, currChunk + numThreads);

        // Consume completions for the previous window
        if (threadNum == 0)
        {
            waitForIoCompletion(&qp->cq, sq, errCount);
            nvm_sq_submit(sq);
        }
        __syncthreads();

        // Move received chunk
        moveBytes(src, bufferOffset * numThreads * chunkSize, dst, currChunk * chunkSize, chunkSize * numThreads);
    
        // Update position and input buffer
        bufferOffset = !bufferOffset;
        currChunk += numThreads;
    }

    // Wait for final buffer to complete
    if (threadNum == 0)
    {
        waitForIoCompletion(&qp->cq, sq, errCount);
    }
    __syncthreads();

    moveBytes(src, bufferOffset * numThreads * chunkSize, dst, currChunk * chunkSize, chunkSize * numThreads);
}



__global__ static
void readSingleBuffered(QueuePair* qp, const uint64_t ioaddr, void* src, void* dst, size_t numChunks, uint64_t* errCount)
{
}



static double launchNvmKernel(const Controller& ctrl, BufferPtr destination, size_t totalChunks, const Settings& settings)
{
    QueuePair queuePair;
    DmaPtr queueMemory = prepareQueuePair(queuePair, ctrl, settings);

    // Set up and prepare queues
    auto deviceQueue = createBuffer(sizeof(QueuePair), settings.cudaDevice);
    auto err = cudaMemcpy(deviceQueue.get(), &queuePair, sizeof(QueuePair), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        throw err;
    }

    const size_t pageSize = ctrl.info.page_size;
    const size_t chunkSize = pageSize * settings.numPages;

    // Create input buffer
    const size_t sourceBufferSize = NVM_PAGE_ALIGN((settings.doubleBuffered + 1) * chunkSize * settings.numThreads, 1UL << 16);
    auto source = createDma(ctrl.ctrl, sourceBufferSize, settings.cudaDevice, settings.adapter, settings.segmentId + 1); // vaddr is a dev ptr

    // We want to count number of errors
    uint64_t* ec = nullptr;
    err = cudaMalloc(&ec, sizeof(uint64_t));
    if (err != cudaSuccess)
    {
        throw err;
    }

    // Launch kernel
    double elapsed = 0;
    try
    {
        Event before, after; 

        before.record(0);
        if (settings.doubleBuffered)
        {
            readDoubleBuffered<<<1, settings.numThreads>>>((QueuePair*) deviceQueue.get(), source->ioaddrs[0], source->vaddr, destination.get(), totalChunks, ec);
        }
        else
        {
            readSingleBuffered<<<1, settings.numThreads>>>((QueuePair*) deviceQueue.get(), source->ioaddrs[0], source->vaddr, destination.get(), totalChunks, ec);
        }
        after.record(0);

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            throw err;
        }

        float msecs = 0;
        cudaEventElapsedTime(&msecs, before.event, after.event);
        elapsed = msecs * 1e3;
    }
    catch (const error& e)
    {
        cudaFree(ec);
        throw e;
    }

    // Check error status
    uint64_t errorCount = 0;
    cudaMemcpy(&errorCount, ec, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(ec);

    if (errorCount != 0)
    {
        fprintf(stderr, "WARNING: There were NVM errors\n");
    }

    return elapsed;
}


static void outputFile(BufferPtr data, size_t size, const char* filename)
{
    auto buffer = createBuffer(size);

    cudaError_t err = cudaMemcpy(buffer.get(), data.get(), size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to copy data from destination: ") + cudaGetErrorString(err));
    }

    FILE* fp = fopen(filename, "wb");
    fwrite(buffer.get(), 1, size, fp);
    fclose(fp);
}



int main(int argc, char** argv)
{
#ifdef __DIS_CLUSTER__
    sci_error_t err;
    SCIInitialize(0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to initialize SISCI: %s\n", SCIGetErrorString(err));
        return 1;
    }
#endif

    Settings settings;
    try
    {
        settings.parseArguments(argc, argv);
    }
    catch (const string& e)
    {
        fprintf(stderr, "%s\n", e.c_str());
        return 1;
    }

    try
    {
        if (settings.blockDevicePath != nullptr)
        {
        }
        else
        {
#ifdef __DIS_CLUSTER__
            Controller ctrl(settings.controllerId, settings.nvmNamespace, settings.adapter, settings.segmentId++);
#else
            int fd = open(settings.controllerPath, O_RDWR | O_NONBLOCK);
            if (fd < 0)
            {
                throw error(strerror(errno));
            }

            close(fd);
#endif
            ctrl.reserveQueues(1);

            const size_t pageSize = ctrl.info.page_size;
            const size_t blockSize = ctrl.ns.lba_data_size;
            const size_t chunkSize = pageSize * settings.numPages;
            const size_t totalChunks = settings.numChunks * settings.numThreads;
            const size_t totalPages = totalChunks * settings.numPages;
            const size_t totalBlocks = NVM_PAGE_TO_BLOCK(pageSize, blockSize, totalPages);

            if (chunkSize > ctrl.info.max_data_size)
            {
                throw error("Chunk size can not be larger than controller data size");
            }
            else if (totalBlocks > ctrl.ns.size)
            {
                throw error("Requesting read size larger than disk size");
            }

            fprintf(stderr, "Number of chunks      : %zu\n", settings.numChunks);
            fprintf(stderr, "Number of pages       : %zu\n", settings.numPages);
            fprintf(stderr, "Number of threads     : %zu\n", settings.numThreads);
            fprintf(stderr, "Total number of pages : %zu\n", totalPages);
            fprintf(stderr, "Total number of blocks: %zu\n", totalBlocks);


            auto outputBuffer = createBuffer(ctrl.info.page_size * totalPages, settings.cudaDevice);

            cudaError_t err = cudaHostRegister((void*) ctrl.ctrl->mm_ptr, NVM_CTRL_MEM_MINSIZE, cudaHostRegisterIoMemory);
            if (err != cudaSuccess)
            {
                throw error(string("Unexpected error while mapping IO memory: ") + cudaGetErrorString(err));
            }

            try
            {
                double usecs = launchNvmKernel(ctrl, outputBuffer, totalChunks, settings);

                fprintf(stderr, "Bandwidth = %.3f MiB/s\n", (totalPages * pageSize) / usecs);

                if (settings.output != nullptr)
                {
                    outputFile(outputBuffer, totalPages * pageSize, settings.output);
                }
            }
            catch (const error& e)
            {
                cudaHostUnregister((void*) ctrl.ctrl->mm_ptr);
                throw e;
            }
            catch (const cudaError_t err)
            {
                cudaHostUnregister((void*) ctrl.ctrl->mm_ptr);
                throw error(string("Unexpected CUDA error: ") + cudaGetErrorString(err));
            }
        }
    }
    catch (const error& e)
    {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }


#ifdef __DIS_CLUSTER__
    SCITerminate();
#endif
    return 0;
}
