#include "queue.h"
#include "buffer.h"
#include "ctrl.h"
#include <stdexcept>
#include <string>
#include <cstdint>
#include <cstddef>
#include <nvm_util.h>



LocalQueue::LocalQueue(const CtrlPtr& ctrl, uint16_t no, size_t depth, size_t pages, uint32_t adapter, uint32_t segmentId)
    : QueuePair(ctrl, no, depth, pages,
            allocateBuffer(*ctrl, (NVM_CQ_PAGES(ctrl->handle, depth) + NVM_SQ_PAGES(ctrl->handle, depth) + depth) * ctrl->pageSize, adapter, segmentId))
{
    nodeId = nvm_dis_node_from_dma(sqMemory.get());
}



GpuQueue::GpuQueue(const CtrlPtr& ctrl, uint16_t no, size_t depth, size_t pages, const GpuPtr& gpu, uint32_t adapter, uint32_t cqSegId, uint32_t sqSegId)
    : QueuePair(ctrl, no, depth, pages,
            allocateBuffer(*ctrl, NVM_CQ_PAGES(ctrl->handle, depth) * ctrl->pageSize, adapter, cqSegId),
            gpu->allocateBufferAndMap(*ctrl, (NVM_SQ_PAGES(ctrl->handle, depth) + depth) * ctrl->pageSize, sqSegId))
    , gpu(gpu)
{
    nodeId = gpu->nodeId;
}



RemoteQueue::RemoteQueue(const CtrlPtr& ctrl, uint16_t no, size_t depth, size_t pages, uint32_t adapter, uint32_t segmentId)
    : QueuePair(ctrl, no, depth, pages,
            allocateBuffer(*ctrl, NVM_CQ_PAGES(ctrl->handle, depth) * ctrl->pageSize, adapter, segmentId),
            deviceBuffer(*ctrl, (NVM_SQ_PAGES(ctrl->handle, depth) + depth) * ctrl->pageSize))
{
    nodeId = nvm_dis_node_from_dma(sqMemory.get());
}

