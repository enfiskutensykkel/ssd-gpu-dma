#ifndef __QUEUE_H__
#define __QUEUE_H__

#include <nvm_types.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "transfer.h"
#include "buffer.h"
#include "ctrl.h"


struct Queue
{
    uint16_t                no;
    BufferPtr               sq_mem;
    BufferPtr               cq_mem;
    nvm_queue_t             sq;
    nvm_queue_t             cq;
    size_t                  depth;
    TransferList            warmups;
    TransferList            transfers;

    Queue(const Controller& ctrl, uint32_t adapter, uint32_t segmentId, uint16_t no, size_t depth, bool remote);
};


typedef std::shared_ptr<Queue> QueuePtr;
typedef std::vector<QueuePtr> QueueList;

#endif
