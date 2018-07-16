#include "queue.h"
#include "buffer.h"
#include "ctrl.h"
#include <stdexcept>
#include <string>
#include <cstdint>
#include <cstddef>



LocalQueue::LocalQueue(const CtrlPtr& ctrl, uint16_t no, size_t depth, size_t pages, uint32_t adapter, uint32_t segmentId)
    : QueuePair(ctrl, no, depth, pages,
            allocateBuffer(*ctrl, (2 + depth) * ctrl->pageSize, adapter, segmentId))
{
}



GpuQueue::GpuQueue(const CtrlPtr& ctrl, uint16_t no, size_t depth, size_t pages, const GpuPtr& gpu, uint32_t adapter, uint32_t cqSegId, uint32_t sqSegId)
    : QueuePair(ctrl, no, depth, pages,
            allocateBuffer(*ctrl, ctrl->pageSize, adapter, cqSegId),
            gpu->allocateBufferAndMap(*ctrl, (depth + 1) * ctrl->pageSize, sqSegId))
    , gpu(gpu)
{
}



RemoteQueue::RemoteQueue(const CtrlPtr& ctrl, uint16_t no, size_t depth, size_t pages, uint32_t adapter, uint32_t segmentId)
    : QueuePair(ctrl, no, depth, pages,
            allocateBuffer(*ctrl, ctrl->pageSize, adapter, segmentId),
            connectBuffer(*ctrl, (depth + 1) * ctrl->pageSize, adapter, no))
{
}

