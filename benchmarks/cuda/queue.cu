#include <cuda.h>
#include "settings.h"
#include "ctrl.h"
#include "buffer.h"
#include "queue.h"
#include <nvm_util.h>
#include <nvm_error.h>
#include <nvm_types.h>
#include <nvm_admin.h>
#include <stdexcept>
#include <string>
#include <cstdint>

using error = std::runtime_error;
using std::string;



__host__ DmaPtr prepareQueuePair(QueuePair& qp, const Controller& ctrl, const Settings& settings)
{
    size_t queueMemSize = ctrl.info.page_size * 2;
    size_t prpListSize = ctrl.info.page_size * settings.numThreads * (settings.doubleBuffered + 1);

    // qmem->vaddr will be already a device pointer after the following call
    auto qmem = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(queueMemSize + prpListSize, 1UL << 16), settings.cudaDevice, settings.adapter, settings.segmentId);

    // Set members
    qp.pageSize = ctrl.info.page_size;
    qp.blockSize = ctrl.ns.lba_data_size;
    qp.nvmNamespace = ctrl.ns.ns_id;
    qp.pagesPerChunk = settings.numPages;
    qp.doubleBuffered = settings.doubleBuffered;
    
    qp.prpList = NVM_DMA_OFFSET(qmem, 2);
    qp.prpListIoAddr = qmem->ioaddrs[2];
    
    // Create completion queue
    int status = nvm_admin_cq_create(ctrl.aq_ref, &qp.cq, 1, qmem->vaddr, qmem->ioaddrs[0]);
    if (!nvm_ok(status))
    {
        throw error(string("Failed to create completion queue: ") + nvm_strerror(status));
    }

    // Get a valid device pointer for CQ doorbell
    void* devicePtr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&devicePtr, (void*) qp.cq.db, 0);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to get device pointer") + cudaGetErrorString(err));
    }
    qp.cq.db = (volatile uint32_t*) devicePtr;

    // Create submission queue
    status = nvm_admin_sq_create(ctrl.aq_ref, &qp.sq, &qp.cq, 1, NVM_DMA_OFFSET(qmem, 1), qmem->ioaddrs[1]);
    if (!nvm_ok(status))
    {
        throw error(string("Failed to create submission queue: ") + nvm_strerror(status));
    }

    // Get a valid device pointer for SQ doorbell
    err = cudaHostGetDevicePointer(&devicePtr, (void*) qp.sq.db, 0);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to get device pointer") + cudaGetErrorString(err));
    }
    qp.sq.db = (volatile uint32_t*) devicePtr;

    return qmem;
}

