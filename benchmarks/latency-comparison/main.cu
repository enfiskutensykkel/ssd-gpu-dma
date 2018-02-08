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
#ifdef __DIS_CLUSTER__
#include <sisci_api.h>
#endif

using error = std::runtime_error;
using std::string;


struct __align__(64) QueuePair
{
    size_t              pageSize;
    size_t              blockSize;
    uint32_t            nvmNamespace;
    size_t              maxDataSize;
    //void*               prpListPtr;
    //uint64_t            prpListAddr;
    nvm_queue_t         sq;
    nvm_queue_t         cq;
};


__device__ void movePage(volatile void* src, void* dst, size_t offset, uint16_t threadNum, size_t pageSize)
{
    uint8_t* source = ((uint8_t*) src) + (pageSize * threadNum);
    uint8_t* destination = ((uint8_t*) dst) + (pageSize * offset);

    for (size_t i = 0; i < pageSize; ++i)
    {
        destination[i] = source[i];
        source[i] = 0;
    }
}



__global__ void readPages(QueuePair* qp, uint64_t ioaddr, volatile void* src, void* dst, size_t numPages, uint64_t* errCount)
{
    const uint16_t numThreads = blockDim.x;
    const uint16_t threadNum = threadIdx.x;

    const size_t blocksPerPage = NVM_PAGE_TO_BLOCK(qp->pageSize, qp->blockSize, 1);
    uint64_t blockOffset = 0;  // FIXME

    if (threadNum == 0)
    {
        *errCount = 0;
    }
    __syncthreads();

    nvm_cmd_t* cmd = nullptr;
    for (size_t currPage = threadNum; currPage < numPages; currPage += numThreads)
    {
        size_t currBlock = NVM_PAGE_TO_BLOCK(qp->pageSize, qp->blockSize, currPage) + blockOffset;
        cmd = nvm_sq_enqueue_n(&qp->sq, cmd, numThreads, threadNum);

        nvm_cmd_header(cmd, NVM_IO_READ, qp->nvmNamespace);
        nvm_cmd_data_ptr(cmd, ioaddr + qp->pageSize * threadNum, 0);
        nvm_cmd_rw_blks(cmd, currBlock, blocksPerPage);

        if (threadNum == 0)
        {
            nvm_sq_submit(&qp->sq);

            for (uint16_t i = 0; i < numThreads; ++i)
            {
                nvm_cpl_t* cpl = nullptr;
                while ((cpl = nvm_cq_dequeue(&qp->cq)) == nullptr);
                nvm_sq_update(&qp->sq);

                if (!NVM_ERR_OK(cpl))
                {
                    *errCount = *errCount + 1;
                }
            }

            nvm_cq_update(&qp->cq);
        }
        //__threadfence();
        __syncthreads();

        movePage(src, dst, currPage, threadNum, qp->pageSize);
    }
}



static void prepareQueuePair(DmaPtr& qmem, QueuePair& qp, const Controller& ctrl, int device, uint32_t numThreads, uint32_t adapter, uint32_t id)
{
    size_t queueMemSize = ctrl.info.page_size * 2;
    //size_t prpListSize = ctrl.info.page_size * numThreads;
    size_t prpListSize = 0;

    // qmem->vaddr will be already a device pointer after the following call
    qmem = createDma(ctrl.ctrl, queueMemSize + prpListSize, device, adapter, id);

    qp.pageSize = ctrl.info.page_size;
    qp.blockSize = ctrl.ns.lba_data_size;
    qp.nvmNamespace = ctrl.ns.ns_id;
    qp.maxDataSize = ctrl.info.max_data_size;
    
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

   // qp.prpListPtr = NVM_DMA_OFFSET(qmem, 2);
   // qp.prpListAddr = qmem->ioaddrs[2];
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




static void use_nvm(const Controller& ctrl, const Settings& settings)
{
    uint32_t sid = settings.segmentId;

    DmaPtr queueMemory;
    QueuePair queuePair;
    prepareQueuePair(queueMemory, queuePair, ctrl, settings.cudaDevice, settings.numThreads, settings.adapter, sid++);

    auto deviceQueue = createBuffer(sizeof(QueuePair), settings.cudaDevice);
    auto err = cudaMemcpy(deviceQueue.get(), &queuePair, sizeof(QueuePair), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        throw err;
    }

    const size_t pageSize = ctrl.info.page_size;
    const size_t blockSize = ctrl.ns.lba_data_size;

    size_t totalPages = NVM_PAGE_ALIGN(settings.numPages * pageSize, pageSize * settings.numThreads) / pageSize;
    size_t totalBlocks = NVM_PAGE_TO_BLOCK(pageSize, blockSize, totalPages);

    fprintf(stderr, "numThreads=%u, totalPages=%zu, totalBlocks=%zu\n",
            settings.numThreads, totalPages, totalBlocks);

    auto destination = createBuffer(pageSize * totalPages, settings.cudaDevice); // this is a host ptr
    
    auto source = createDma(ctrl.ctrl, pageSize * settings.numThreads, settings.cudaDevice, settings.adapter, sid++); // vaddr is a dev ptr

    uint64_t* ec = nullptr;
    err = cudaMalloc(&ec, sizeof(uint64_t));
    if (err != cudaSuccess)
    {
        throw err;
    }

    readPages<<<1, settings.numThreads>>>((QueuePair*) deviceQueue.get(), source->ioaddrs[0], source->vaddr, destination.get(), totalPages, ec);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        cudaFree(ec);
        throw err;
    }

    uint64_t errorCount = 0;
    cudaMemcpy(&errorCount, ec, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(ec);

    fprintf(stderr, "ec: %lx\n", errorCount);

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
