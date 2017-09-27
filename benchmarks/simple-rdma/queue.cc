#include "queue.h"
#include "manager.h"
#include <nvm_util.h>
#include <nvm_types.h>
#include <nvm_dma.h>
#include <nvm_rpc.h>
#include <nvm_queue.h>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>


QueueManager::QueueManager(ManagerPtr manager, int cq_id, int sq_id_start, int num_queues)
    : queue_map(nullptr)
{
    int err;
    nvm_rpc_t ref;

    void* queue_memory = nullptr;
    size_t queue_memory_size = (1 + num_queues) * manager->ctrl->page_size;

    if ((err = nvm_rpc_bind_local(&ref, manager->manager)) != 0)
    {
        throw std::runtime_error("Could not bind local reference");
    }

    err = posix_memalign(&queue_memory, manager->ctrl->page_size, queue_memory_size);
    if (err != 0)
    {
        nvm_rpc_unbind(ref);
        throw std::runtime_error("Could not allocate queue memory");
    }

    err = nvm_dma_window_host_map(&queue_map, manager->ctrl, queue_memory, queue_memory_size);
    if (err != 0)
    {
        free(queue_memory);
        nvm_rpc_unbind(ref);
        throw std::runtime_error("Failed to map queue memory");
    }

    err = nvm_rpc_cq_create(&completion_queue, ref, manager->ctrl, cq_id, queue_memory, queue_map->ioaddrs[0]);
    if (err != 0)
    {
        nvm_dma_window_free(queue_map);
        free(queue_memory);
        nvm_rpc_unbind(ref);
        throw std::runtime_error("Failed to create CQ");
    }

    for (int i = 0; i < num_queues; ++i)
    {
        nvm_queue_t sq;

        uint16_t id = sq_id_start + i;
        void* ptr = DMA_VADDR(queue_memory, queue_map->page_size, 1 + i);
        uint64_t addr = queue_map->ioaddrs[1 + i];

        err = nvm_rpc_sq_create(&sq, ref, manager->ctrl, &completion_queue, id, ptr, addr);
        if (err != 0)
        {
            nvm_dma_window_free(queue_map);
            free(queue_memory);
            nvm_rpc_unbind(ref);
            throw std::runtime_error("Failed to create SQ number " + std::to_string(i));
        }

        submission_queues.push_back(sq);
    }

    nvm_rpc_unbind(ref);
}


QueueManager::~QueueManager()
{
    void* ptr = queue_map->vaddr;
    nvm_dma_window_free(queue_map);
    free(ptr);
}

