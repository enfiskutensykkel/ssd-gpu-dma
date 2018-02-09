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
#include "ctrl.h"
#include "buffer.h"
#include "settings.h"
#include "event.h"
#ifdef __DIS_CLUSTER__
#include <sisci_api.h>
#endif

using error = std::runtime_error;
using std::string;


struct __align__(64) QueuePair
{
    uint32_t            pageSize;
    uint32_t            blockSize;
    uint32_t            nvmNamespace;
    uint32_t            pagesPerChunk;
    uint16_t            bufferLevel;
    void*               prpList;
    uint64_t            prpListIoAddr;
    nvm_queue_t         sq;
    nvm_queue_t         cq;
};


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

    const uint16_t blocksPerChunk = NVM_PAGE_TO_BLOCK(pageSize, blockSize, chunkPages);
    const uint64_t currBlock = NVM_PAGE_TO_BLOCK(pageSize, blockSize, (currChunk + threadNum) * chunkPages);

    // Prepare PRP list building
    void* prpList = NVM_PTR_OFFSET(qp->prpList, pageSize, threadOffset);
    uint64_t prpListAddr = NVM_ADDR_OFFSET(qp->prpListIoAddr, pageSize, threadOffset);

    uint64_t addrs[0x1000 / sizeof(uint64_t)]; // FIXME: hack
    for (uint32_t page = 0; page < chunkPages; ++page)
    {
        addrs[page] = NVM_ADDR_OFFSET(ioaddr, pageSize, chunkPages * threadOffset + page);
    }

    nvm_cmd_t* cmd = nvm_sq_enqueue_n(&qp->sq, last, numThreads, threadNum);

    nvm_cmd_header(cmd, NVM_IO_READ, nvmNamespace);
    nvm_cmd_data(cmd, pageSize, chunkPages, prpList, prpListAddr, addrs);
    nvm_cmd_rw_blks(cmd, currBlock + blockOffset, blocksPerChunk);

    __threadfence();
    return cmd;
}



__global__ void readPages(QueuePair* qp, const uint64_t ioaddr, void* src, void* dst, size_t numChunks, uint64_t* errCount)
{
    const uint16_t numThreads = blockDim.x;
    const uint16_t threadNum = threadIdx.x;
    const uint16_t bufferLevel = 2;//qp->bufferLevel;
    const uint32_t pageSize = qp->pageSize;
    const size_t chunkSize = qp->pagesPerChunk * pageSize;
    nvm_queue_t* sq = &qp->sq;

    uint64_t blockOffset = 0; // TODO: Fix this

    uint32_t currChunk = 0;
    uint16_t bufferOffset = 0;

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
        last = prepareChunk(qp, last, ioaddr, (bufferOffset + 1) % bufferLevel, blockOffset, currChunk + numThreads);

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
        bufferOffset = (bufferOffset + 1) % bufferLevel;
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



static void prepareQueuePair(DmaPtr& qmem, QueuePair& qp, const Controller& ctrl, const Settings& settings)
{
    size_t queueMemSize = ctrl.info.page_size * 2;
    size_t prpListSize = ctrl.info.page_size * settings.numThreads * settings.bufferLevel;

    // qmem->vaddr will be already a device pointer after the following call
    qmem = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(queueMemSize + prpListSize, 1UL << 16), settings.cudaDevice, settings.adapter, settings.segmentId);

    qp.pageSize = ctrl.info.page_size;
    qp.blockSize = ctrl.ns.lba_data_size;
    qp.nvmNamespace = ctrl.ns.ns_id;
    qp.pagesPerChunk = settings.numPages;
    qp.bufferLevel = settings.bufferLevel;
    
    int status = nvm_admin_cq_create(ctrl.aq_ref, &qp.cq, 1, qmem->vaddr, qmem->ioaddrs[0]);
    if (!nvm_ok(status))
    {
        throw error(string("Failed to create completion queue: ") + nvm_strerror(status));
    }

    void* devicePtr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&devicePtr, (void*) qp.cq.db, 0);
    if (err != cudaSuccess)
    {
        throw err;
    }
    qp.cq.db = (volatile uint32_t*) devicePtr;

    status = nvm_admin_sq_create(ctrl.aq_ref, &qp.sq, &qp.cq, 1, NVM_DMA_OFFSET(qmem, 1), qmem->ioaddrs[1]);
    if (!nvm_ok(status))
    {
        throw error(string("Failed to create submission queue: ") + nvm_strerror(status));
    }

    err = cudaHostGetDevicePointer(&devicePtr, (void*) qp.sq.db, 0);
    if (err != cudaSuccess)
    {
        throw err;
    }
    qp.sq.db = (volatile uint32_t*) devicePtr;

    qp.prpList = NVM_DMA_OFFSET(qmem, 2);
    qp.prpListIoAddr = qmem->ioaddrs[2];
}



static void verify(BufferPtr data, size_t size, const char* filename)
{
    auto buffer = createBuffer(size);

    cudaError_t err = cudaMemcpy(buffer.get(), data.get(), size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to copy data from destination: ") + cudaGetErrorString(err));
    }

    // TODO: open filename for read and compare byte by byte

    FILE* fp = fopen(filename, "wb");
    fwrite(buffer.get(), 1, size, fp);
    fclose(fp);
}




static void use_nvm(const Controller& ctrl, const Settings& settings) // TODO take destination as argument
{
    DmaPtr queueMemory;
    QueuePair queuePair;
    prepareQueuePair(queueMemory, queuePair, ctrl, settings);

    auto deviceQueue = createBuffer(sizeof(QueuePair), settings.cudaDevice);
    auto err = cudaMemcpy(deviceQueue.get(), &queuePair, sizeof(QueuePair), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        throw err;
    }

    const size_t pageSize = ctrl.info.page_size;
    const size_t blockSize = ctrl.ns.lba_data_size;
    const size_t chunkSize = pageSize * settings.numPages;

    if (chunkSize > ctrl.info.max_data_size)
    {
        throw error("Chunk size can not be larger than controller data size");
    }

    size_t totalChunks = settings.numChunks * settings.numThreads;
    size_t totalPages = totalChunks * settings.numPages;
    size_t totalBlocks = NVM_PAGE_TO_BLOCK(pageSize, blockSize, totalPages);

    fprintf(stderr, "numThreads=%u, numChunks=%zu, pagesPerChunk=%zu, totalChunks=%zu, totalPages=%zu, totalBlocks=%zu\n",
            settings.numThreads, settings.numChunks, settings.numPages, totalChunks, totalPages, totalBlocks);

    auto destination = createBuffer(pageSize * totalPages, settings.cudaDevice); // this is a host ptr
    
    const size_t sourceBufferSize = NVM_PAGE_ALIGN(settings.bufferLevel * chunkSize * settings.numThreads, 1UL << 16);
    auto source = createDma(ctrl.ctrl, sourceBufferSize, settings.cudaDevice, settings.adapter, settings.segmentId + 1); // vaddr is a dev ptr

    Event before, after; 

    uint64_t* ec = nullptr;
    err = cudaMalloc(&ec, sizeof(uint64_t));
    if (err != cudaSuccess)
    {
        throw err;
    }

    try
    {
        before.record(0);
        readPages<<<1, settings.numThreads>>>((QueuePair*) deviceQueue.get(), source->ioaddrs[0], source->vaddr, destination.get(), totalChunks, ec);
        after.record(0);

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            throw err;
        }
    }
    catch (const error& e)
    {
        cudaFree(ec);
        throw e;
    }

    float msecs = 0;
    cudaEventElapsedTime(&msecs, before.event, after.event);
    double usecs = (msecs * 1e3);

    fprintf(stderr, "BW=%.3f MiB/s\n", (totalPages * pageSize) / usecs);

    uint64_t errorCount = 0;
    cudaMemcpy(&errorCount, ec, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(ec);

    fprintf(stderr, "ec: 0x%lx\n", errorCount);

    if (settings.verify != nullptr)
    {
        verify(destination, totalPages * pageSize, settings.verify);
    }
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
        if (argc != 2)
        {
            Controller ctrl(settings.controllerId, settings.nvmNamespace, settings.adapter, settings.segmentId++);
            ctrl.reserveQueues(1);

            cudaError_t err = cudaHostRegister((void*) ctrl.ctrl->mm_ptr, NVM_CTRL_MEM_MINSIZE, cudaHostRegisterIoMemory);
            if (err != cudaSuccess)
            {
                throw err;
            }

            use_nvm(ctrl, settings);

            cudaHostUnregister((void*) ctrl.ctrl->mm_ptr);
        }
        else
        {
            //use_fd(grid, block, argv[1]);
        }
    }
    catch (const cudaError_t err)
    {
        fprintf(stderr, "Unexpected CUDA error: %s\n", cudaGetErrorString(err));
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
