#ifndef __SIMPLE_RDMA_QUEUE_H__
#define __SIMPLE_RDMA_QUEUE_H__

#include <nvm_types.h>
#include <vector>
#include <memory>
#include "manager.h"

struct QueueManager;
typedef std::shared_ptr<QueueManager> QueueManagerPtr;


struct QueueManager
{
    nvm_dma_t                   queue_map;
    nvm_queue_t                 completion_queue;
    std::vector<nvm_queue_t>    submission_queues;

    QueueManager(ManagerPtr manager, int cq_id, int sq_id_start, int num_queues);
    ~QueueManager() noexcept;
};

#endif
