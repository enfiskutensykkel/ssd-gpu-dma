#include "queue.h"
#include "buffer.h"
#include "ctrl.h"
#include <nvm_admin.h>
#include <nvm_types.h>
#include <nvm_util.h>
#include <nvm_error.h>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cstring>

using std::string;
using error = std::runtime_error;



QueuePair::QueuePair()
    : no(0)
    , depth(0)
    , pages(0)
{
    throw std::logic_error("SmartIO not supported");
}



QueuePair::QueuePair(const CtrlPtr& ctrl, uint16_t no, size_t depth, size_t pages, DmaPtr cqPtr, DmaPtr sqPtr)
    : no(no)
    , depth(depth)
    , pages(std::max(pages, (size_t) 1))
    , controller(ctrl)
    , cqMemory(cqPtr)
    , sqMemory(sqPtr)
{
    if (depth > ctrl->maxEntries)
    {
        throw error("Queue depth is greater than maximum queue size");
    }

    if (this->pages * ctrl->pageSize > ctrl->chunkSize)
    {
        throw error("Chunk size is greater than maximum data transfer size");
    }

    if (sqPtr->n_ioaddrs < NVM_SQ_PAGES(ctrl->handle, depth) + depth)
    {
        throw error("Submission queue memory is not large enough for queue depth and PRP lists");
    }

    if (cqPtr->n_ioaddrs < NVM_CQ_PAGES(ctrl->handle, depth))
    {
        throw error("Completion queue memory is not large enough for queue depth");
    }

    nvm_aq_ref ref = controller->adminReference();

    memset(cqMemory->vaddr, 0, NVM_CQ_PAGES(ctrl->handle, depth) * controller->pageSize);
    int status = nvm_admin_cq_create(ref, &cq, no, cqMemory.get(), 0, depth);
    if (!nvm_ok(status))
    {
        throw error("Failed to create completion queue: " + string(nvm_strerror(status)));
    }

    memset(sqMemory->vaddr, 0, NVM_SQ_PAGES(ctrl->handle, depth) * controller->pageSize);
    status = nvm_admin_sq_create(ref, &sq, &cq, no, sqMemory.get(), 0, depth);
    if (!nvm_ok(status))
    {
        throw error("Failed to create submission queue: " + string(nvm_strerror(status)));
    }
}



QueuePair::QueuePair(const CtrlPtr& ctrl, uint16_t no, size_t depth, size_t pages, DmaPtr queueMemory)
    : no(no)
    , depth(depth)
    , pages(std::max(pages, (size_t) 1))
    , controller(ctrl)
    , cqMemory(nullptr)
    , sqMemory(queueMemory)
{
    if (depth > ctrl->maxEntries)
    {
        throw error("Queue depth is greater than maximum queue size");
    }

    if (this->pages * ctrl->pageSize > ctrl->chunkSize)
    {
        throw error("Chunk size is greater than maximum data transfer size");
    }

    if (queueMemory->n_ioaddrs < NVM_CQ_PAGES(ctrl->handle, depth) + NVM_SQ_PAGES(ctrl->handle, depth) + this->depth)
    {
        throw error("Queue memory is not large enough for queue depth and PRP lists");
    }

    nvm_aq_ref ref = controller->adminReference();

    size_t cqOffset = NVM_SQ_PAGES(ctrl->handle, this->depth) + this->depth;
    void* cqPtr = NVM_DMA_OFFSET(queueMemory, cqOffset);

    memset(cqPtr, 0, NVM_CQ_PAGES(ctrl->handle, this->depth) * ctrl->pageSize);
    int status = nvm_admin_cq_create(ref, &cq, no, queueMemory.get(), cqOffset, this->depth);
    if (!nvm_ok(status))
    {
        throw error("Failed to create completion queue: " + string(nvm_strerror(status)));
    }

    memset(queueMemory->vaddr, 0, NVM_SQ_PAGES(ctrl->handle, this->depth) * controller->pageSize);
    status = nvm_admin_sq_create(ref, &sq, &cq, no, queueMemory.get(), 0, this->depth);
    if (!nvm_ok(status))
    {
        throw error("Failed to create submission queue: " + string(nvm_strerror(status)));
    }
}



QueuePair::~QueuePair()
{
    nvm_aq_ref ref = controller->adminReference();
    nvm_admin_sq_delete(ref, &sq, &cq);
    nvm_admin_cq_delete(ref, &cq);
}



LocalQueue::LocalQueue(const CtrlPtr& ctrl, uint16_t no, size_t depth, size_t pages)
    : QueuePair(
            ctrl, 
            no, 
            depth, 
            pages, 
            allocateBuffer(*ctrl, (NVM_CQ_PAGES(ctrl->handle, depth) + NVM_SQ_PAGES(ctrl->handle, depth) + depth) * ctrl->pageSize)
            )
{
}



#ifndef __DIS_CLUSTER__
LocalQueue::LocalQueue(const CtrlPtr& ctrl, uint16_t no, size_t depth, size_t pages, uint32_t, uint32_t)
    : QueuePair(
            ctrl, 
            no, 
            depth, 
            pages, 
            allocateBuffer(*ctrl, (NVM_CQ_PAGES(ctrl->handle, depth) + NVM_SQ_PAGES(ctrl->handle, depth) + depth) * ctrl->pageSize)
            )
{
}
#endif


string LocalQueue::type() const 
{
    return "host";
}



string RemoteQueue::type() const
{
    return "remote";
}



string GpuQueue::type() const
{
    return "gpu";
}



#ifndef __DIS_CLUSTER__
GpuQueue::GpuQueue(const CtrlPtr&, uint16_t, size_t, size_t, const GpuPtr&, uint32_t, uint32_t, uint32_t)
{
    throw std::logic_error("SmartIO not supported");
}
#endif



#ifndef __DIS_CLUSTER__
RemoteQueue::RemoteQueue(const CtrlPtr&, uint16_t, size_t, size_t, uint32_t, uint32_t)
{
    throw std::logic_error("SmartIO not supported");
}
#endif
