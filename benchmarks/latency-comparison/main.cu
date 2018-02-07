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


//static void prepareQueue(nvm_queue_t* q)
//{
//    cudaError_t err = cudaHostRegister((void*) q->db, sizeof(uint32_t), cudaHostRegisterIoMemory);
//    if (err != cudaSuccess)
//    {
//        throw error(string("Failed to register IO memory: ") + cudaGetErrorString(err));
//    }
//
//    void* devicePtr = nullptr;
//    err = cudaHostGetDevicePointer(&devicePtr, (void*) q->db, 0);
//    if (err != cudaSuccess)
//    {
//        cudaHostUnregister((void*) q->db);
//        throw error(string("Failed to get device pointer: ") + cudaGetErrorString(err));
//    }
//
//    err = cudaMemset((void*) q->vaddr, 0, q->max_entries * q->entry_size);
//    if (err != cudaSuccess)
//    {
//        cudaHostUnregister((void*) q->db);
//        throw error(string("Failed to reset queue memory: ") + cudaGetErrorString(err));
//    }
//
//    q->db = (volatile uint32_t*) devicePtr;
//}
//
//
//static void createQueuePair(QueuePair* deviceMemory, const Controller& ctrl)
//{
//}
//
//
//
//
//
//
//__global__ void moveBlocks(nvm_queue_t* sq, nvm_queue_t* cq, void* dmaBuffer, uint64_t ioaddr, void* destination, 
//        
//        
//        
//        
//        
//        uint64_t ioaddr, void* dst, void* src, uint32_t numBlocks, size_t blockSize, uint32_t ns, uint64_t* ec)
//{
//    const uint16_t threadCount = blockDim.x * gridDim.x;
//    const uint16_t threadId = blockDim.x * blockIdx.x + threadIdx.x;
//
//    nvm_cmd_t* cmd;
//    while ((cmd = nvm_sq_enqueue_n(sq, threadCount, threadId)) == nullptr);
//
//    nvm_cmd_header(cmd, NVM_IO_READ, ns);
//    //nvm_cmd_data_ptr(cmd, ioaddr + blockSize, 0); // TODO work on page sizes instead
//    //nvm_cmd_rw_blks(cmd, i + threadId, 1);
//    nvm_cmd_data_ptr(cmd, ioaddr + (threadId * 8), 0);
//    nvm_cmd_rw_blks(cmd, 0, 8);
//
//    // Warp divergence, not good but unavoidable
//    __syncthreads();
//    //        if (threadId == 0)
//    //        {
//    //            nvm_sq_submit(sq);
//    //
//    //            for (uint32_t ncpls = 0; ncpls < threadCount; ++ncpls)
//    //            {
//    //                while (nvm_cq_dequeue(cq) == nullptr);
//    //                nvm_sq_update(sq);
//    //                nvm_cq_update(cq);
//    //            }
//    //        }
//    __syncthreads();
//
//    for (size_t byte = 0; byte < blockSize; ++byte)
//    {
//        char* dptr = ((char*) dst) + blockSize * (i + threadId);
//        char* sptr = ((char*) src) + blockSize * threadId;
//
//        dptr[byte] = sptr[byte];
//    }
//}
//



struct __align__(64) QueuePair
{
    size_t              pageSize;
    size_t              blockSize;
    uint32_t            nvmNamespace;
    size_t              maxDataSize;
    void*               prpListPtr;
    uint64_t            prpListAddr;
    nvm_queue_t         sq;
    nvm_queue_t         cq;
};





static void prepareQueuePair(DmaPtr& qmem, QueuePair& qp, const Controller& ctrl, int device, uint32_t numThreads, uint32_t adapter, uint32_t id)
{
    size_t queueMemSize = ctrl.info.page_size * 2;
    size_t prpListSize = ctrl.info.page_size * numThreads;

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

    qp.prpListPtr = NVM_DMA_OFFSET(qmem, 2);
    qp.prpListAddr = qmem->ioaddrs[2];
}




static void use_nvm(const Controller& ctrl, const Settings& settings)
{
    uint32_t sid = settings.segmentId;

    DmaPtr queueMemory;
    QueuePair queuePair;
    prepareQueuePair(queueMemory, queuePair, ctrl, settings.cudaDevice, settings.numThreads, settings.adapter, sid++);

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
