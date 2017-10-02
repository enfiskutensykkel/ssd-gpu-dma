#include <nvm_util.h>
#include <nvm_types.h>
#include <nvm_rpc.h>
#include <stdexcept>
#include <vector>
#include <memory>
#include "queue.h"
#include "dma.h"


using std::runtime_error;


void createQueues(nvm_rpc_t rpc, nvm_ctrl_t ctrl, DmaPtr queueMem, QueueList& queues)
{
    uint16_t numQueues = (*queueMem)->n_ioaddrs - 1;

    if (numQueues == 0)
    {
        throw runtime_error("Invalid argument");
    }

    uint16_t cqs = 1;
    uint16_t sqs = numQueues;
    int err = nvm_rpc_request_num_queues(rpc, &cqs, &sqs);
    if (err != 0)
    {
        throw runtime_error("Failed to set number of SQs");
    }

    if (sqs < numQueues)
    {
        throw runtime_error("Requested more queues than available");
    }

    nvm_queue_t cq;
    void* ptr = (*queueMem)->vaddr;
    size_t page_size = (*queueMem)->page_size;
    uint64_t* ioaddrs = (*queueMem)->ioaddrs;

    err = nvm_rpc_cq_create(&cq, rpc, ctrl, 1, DMA_VADDR(ptr, page_size, 0), ioaddrs[0]);
    if (err != 0)
    {
        throw runtime_error("Failed to create CQ");
    }

    queues.push_back(cq);

    for (uint16_t i = 0; i < numQueues; ++i)
    {
        nvm_queue_t sq;
        err = nvm_rpc_sq_create(&sq, rpc, ctrl, &cq, 1 + i, DMA_VADDR(ptr, page_size, 1 + i), ioaddrs[1 + i]);
        if (err != 0)
        {
            throw runtime_error("Failed to create SQ");
        }
        queues.push_back(sq);
    }
}

