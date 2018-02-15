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
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
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
void moveKernel(void* src, void* dst, size_t chunkSize)
{
    const uint16_t numThreads = blockDim.x;
    moveBytes(src, 0, dst, 0, chunkSize * numThreads);
}



__host__ static inline
void launchMoveKernel(size_t pageSize, void* input, void* src, void* dst, size_t currChunk, const Settings& settings)
{
    const auto numPages = settings.numPages;
    const auto numThreads = settings.numThreads;
    const auto chunkSize = pageSize * numPages;

    void* dstPtr = (void*) (((unsigned char*) dst) + chunkSize * currChunk);
    void* inputPtr = (void*) (((unsigned char*) input) + chunkSize * currChunk);

    cudaError_t err = cudaMemcpyAsync(src, inputPtr, chunkSize * numThreads, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        throw err;
    }

    moveKernel<<<1, numThreads>>>(src, dstPtr, chunkSize);
}



static double launchMoveKernelLoop(void* fileMap, BufferPtr destination, size_t pageSize, const Settings& settings)
{
    const size_t chunkSize = pageSize * settings.numPages;
    const size_t numThreads = settings.numThreads;
    const size_t totalChunks = settings.numChunks * numThreads;

    const size_t sourceBufferSize = NVM_PAGE_ALIGN(chunkSize * numThreads, 1UL << 16);
    auto source = createBuffer(sourceBufferSize, settings.cudaDevice);

    Event before, after;

    before.record(0);
    for (size_t currChunk = 0; currChunk < totalChunks; currChunk += numThreads)
    {
        launchMoveKernel(pageSize, fileMap, source.get(), destination.get(), currChunk, settings);
    }
    after.record(0);

    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        throw err;
    }

    float msecs = 0.0;
    cudaEventElapsedTime(&msecs, before.event, after.event);
    double elapsed = msecs * 1e3;

    return elapsed;
}



__global__ static
void readDoubleBuffered(QueuePair* qp, const uint64_t ioaddr, void* src, void* dst, size_t numChunks, uint64_t startBlock, uint64_t* errCount)
{
    const uint16_t numThreads = blockDim.x;
    const uint16_t threadNum = threadIdx.x;
    const uint32_t pageSize = qp->pageSize;
    const size_t chunkSize = qp->pagesPerChunk * pageSize;
    nvm_queue_t* sq = &qp->sq;

    uint64_t blockOffset = startBlock;

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
void readSingleBuffered(QueuePair* qp, const uint64_t ioaddr, void* src, void* dst, size_t numChunks, uint64_t startBlock, uint64_t* errCount)
{
    const uint16_t numThreads = blockDim.x;
    const uint16_t threadNum = threadIdx.x;
    const uint32_t pageSize = qp->pageSize;
    const size_t chunkSize = qp->pagesPerChunk * pageSize;
    nvm_queue_t* sq = &qp->sq;

    uint64_t blockOffset = startBlock;

    uint32_t currChunk = 0;

    nvm_cmd_t* cmd = nullptr;

    if (threadNum == 0)
    {
        *errCount = 0;
    }
    __syncthreads();

    while (currChunk < numChunks)
    {
        // Prepare in advance next chunk
        cmd = prepareChunk(qp, cmd, ioaddr, 0, blockOffset, currChunk);

        // Consume completions for the previous window
        if (threadNum == 0)
        {
            nvm_sq_submit(sq);
            waitForIoCompletion(&qp->cq, sq, errCount);
        }
        __syncthreads();

        // Move received chunk
        moveBytes(src, 0, dst, currChunk * chunkSize, chunkSize * numThreads);
    
        // Update position and input buffer
        currChunk += numThreads;
    }
}



static double launchNvmKernel(const Controller& ctrl, BufferPtr destination, const Settings& settings)
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
    const size_t totalChunks = settings.numChunks * settings.numThreads;

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
            readDoubleBuffered<<<1, settings.numThreads>>>((QueuePair*) deviceQueue.get(), source->ioaddrs[0], source->vaddr, destination.get(), totalChunks, settings.startBlock, ec);
        }
        else
        {
            readSingleBuffered<<<1, settings.numThreads>>>((QueuePair*) deviceQueue.get(), source->ioaddrs[0], source->vaddr, destination.get(), totalChunks, settings.startBlock, ec);
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
    catch (const cudaError_t err)
    {
        cudaFree(ec);
        throw err;
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


static int useBlockDevice(const Settings& settings)
{
    int fd = open(settings.blockDevicePath, O_RDONLY);
    if (fd < 0)
    {
        fprintf(stderr, "Failed to open block device: %s\n", strerror(errno));
        return 1;
    }

    const size_t pageSize = sysconf(_SC_PAGESIZE);
    const size_t blockSize = 512; // FIXME: specify this from command line
    const size_t totalChunks = settings.numChunks * settings.numThreads;
    const size_t totalPages = totalChunks * settings.numPages;

    fprintf(stderr, "Controller page size  : %zu B\n", pageSize);
    fprintf(stderr, "Assumed block size    : %zu B\n", blockSize);
    fprintf(stderr, "Number of threads     : %zu\n", settings.numThreads);
    fprintf(stderr, "Chunks per thread     : %zu\n", settings.numChunks);
    fprintf(stderr, "Pages per chunk       : %zu\n", settings.numPages);
    fprintf(stderr, "Total number of pages : %zu\n", totalPages);
    fprintf(stderr, "Double buffering      : %s\n", settings.doubleBuffered ? "yes" : "no");

    void* ptr = mmap(nullptr, totalPages * pageSize, PROT_READ, MAP_FILE | MAP_PRIVATE, fd, settings.startBlock * blockSize);
    if (ptr == nullptr)
    {
        close(fd);
        fprintf(stderr, "Failed to memory map block device: %s\n", strerror(errno));
        return 1;
    }

    try
    {
        auto outputBuffer = createBuffer(totalPages * pageSize);

        double usecs = launchMoveKernelLoop(ptr, outputBuffer, pageSize, settings);

        fprintf(stdout, "Time elapsed: %.3f µs\n", usecs);
        fprintf(stdout, "Bandwidth   : %.3f MiB/s\n", (totalPages * pageSize) / usecs);

        if (settings.output != nullptr)
        {
            outputFile(outputBuffer, totalPages * pageSize, settings.output);
        }
    }
    catch (const cudaError_t err)
    {
        munmap(ptr, totalPages * pageSize);
        close(fd);
        fprintf(stderr, "Unexpected CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    catch (const error& e)
    {
        munmap(ptr, totalPages * pageSize);
        close(fd);
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }

    munmap(ptr, totalPages * pageSize);
    close(fd);
    return 0;
}



int main(int argc, char** argv)
{
    Settings settings;
    try
    {
        settings.parseArguments(argc, argv);
    }
    catch (const string& e)
    {
        fprintf(stderr, "%s\n", e.c_str());
        fprintf(stderr, "%s\n", Settings::usageString(argv[0]).c_str());
        return 1;
    }

    if (settings.blockDevicePath != nullptr)
    {
        return useBlockDevice(settings);
    }

#ifdef __DIS_CLUSTER__
    sci_error_t err;
    SCIInitialize(0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to initialize SISCI: %s\n", SCIGetErrorString(err));
        return 1;
    }

    sci_desc_t sd;
    SCIOpen(&sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to open SISCI descriptor: %s\n", SCIGetErrorString(err));
        return 1;
    }

    SCIRegisterPCIeRequester(sd, settings.adapter, settings.bus, settings.devfn, SCI_FLAG_PCIE_REQUESTER_GLOBAL, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to register PCI requester: %s\n", SCIGetErrorString(err));
        SCIClose(sd, 0, &err);
        return 1;
    }
    sleep(1); // FIXME: Hack due to race condition in SmartIO
#endif

    try
    {
#ifdef __DIS_CLUSTER__
        Controller ctrl(settings.controllerId, settings.nvmNamespace, settings.adapter, settings.segmentId++);
#else
        Controller ctrl(settings.controllerPath, settings.nvmNamespace);
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

        fprintf(stderr, "Controller page size  : %zu B\n", pageSize);
        fprintf(stderr, "Namespace block size  : %zu B\n", blockSize);
        fprintf(stderr, "Number of threads     : %zu\n", settings.numThreads);
        fprintf(stderr, "Chunks per thread     : %zu\n", settings.numChunks);
        fprintf(stderr, "Pages per chunk       : %zu\n", settings.numPages);
        fprintf(stderr, "Total number of pages : %zu\n", totalPages);
        fprintf(stderr, "Total number of blocks: %zu\n", totalBlocks);
        fprintf(stderr, "Double buffering      : %s\n", settings.doubleBuffered ? "yes" : "no");

        auto outputBuffer = createBuffer(ctrl.info.page_size * totalPages, settings.cudaDevice);

        cudaError_t err = cudaHostRegister((void*) ctrl.ctrl->mm_ptr, NVM_CTRL_MEM_MINSIZE, cudaHostRegisterIoMemory);
        if (err != cudaSuccess)
        {
            throw error(string("Unexpected error while mapping IO memory: ") + cudaGetErrorString(err));
        }

        try
        {
            double usecs = launchNvmKernel(ctrl, outputBuffer, settings);

            fprintf(stdout, "Time elapsed: %.3f µs\n", usecs);
            fprintf(stdout, "Bandwidth   : %.3f MiB/s\n", (totalPages * pageSize) / usecs);

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
    catch (const error& e)
    {
#ifdef __DIS_CLUSTER__
        SCIUnregisterPCIeRequester(sd, settings.adapter, settings.bus, settings.devfn, 0, &err);
        SCIClose(sd, 0, &err);
        SCITerminate();
#endif
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }

#ifdef __DIS_CLUSTER__
    SCIUnregisterPCIeRequester(sd, settings.adapter, settings.bus, settings.devfn, 0, &err);
    SCIClose(sd, 0, &err);
    SCITerminate();
#endif
    return 0;
}
