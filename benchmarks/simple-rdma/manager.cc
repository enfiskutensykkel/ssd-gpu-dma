#include "manager.h"
#include <nvm_util.h>
#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_dma.h>
#include <nvm_manager.h>
#include <nvm_rpc.h>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>


Manager::Manager(uint32_t ctrl_id)
    : ctrl(nullptr)
    , queue_map(nullptr)
    , manager(nullptr)
{
    int err;

    if ((err = nvm_ctrl_init(&ctrl, ctrl_id)) != 0)
    {
        throw std::runtime_error("Could not acquire controller reference: " 
                + std::string(std::strerror(err)));
    }

    void* queue_memory = nullptr;
    size_t queue_memory_size = 2 * ctrl->page_size + 1 * 0x1000;

    err = posix_memalign(&queue_memory, ctrl->page_size, queue_memory_size);
    if (err != 0)
    {
        nvm_ctrl_free(ctrl);
        throw std::runtime_error("Could not allocate admin queue memory");
    }

    err = nvm_dma_window_host_map(&queue_map, ctrl, queue_memory, queue_memory_size);
    if (err != 0)
    {
        free(queue_memory);
        nvm_ctrl_free(ctrl);
        throw std::runtime_error("Could not map queue memory");
    }

    if ((err = nvm_manager_register(&manager, ctrl, queue_map)) != 0)
    {
        nvm_dma_window_free(queue_map);
        free(queue_memory);
        nvm_ctrl_free(ctrl);
        throw std::runtime_error("Failed to reset controller: "
                + std::string(std::strerror(err)));
    }

    void* info_ptr = DMA_VADDR(queue_map->vaddr, queue_map->page_size, 2);
    uint64_t info_addr = queue_map->ioaddrs[2];

    nvm_rpc_t reference;
    if ((err = nvm_rpc_bind_local(&reference, manager)) != 0)
    {
        nvm_manager_unregister(manager);
        nvm_dma_window_free(queue_map);
        free(queue_memory);
        nvm_ctrl_free(ctrl);
        throw std::runtime_error("Failed to get controller information: "
                + std::string(std::strerror(err)));
    }

    err = nvm_rpc_ctrl_info(&ctrl_info, reference, ctrl, info_ptr, info_addr);
    if (err != 0)
    {
        nvm_rpc_unbind(reference);
        nvm_manager_unregister(manager);
        nvm_dma_window_free(queue_map);
        free(queue_memory);
        nvm_ctrl_free(ctrl);
        throw std::runtime_error("Failed to get controller information: "
                + std::string(std::strerror(err)));
    }

    nvm_rpc_unbind(reference);
}


Manager::~Manager()
{
    void* ptr = queue_map->vaddr;
    nvm_manager_unregister(manager);
    nvm_dma_window_free(queue_map);
    free(ptr);
    nvm_ctrl_free(ctrl);
}


size_t Manager::get_blk_size(uint32_t ns_id) const
{
    nvm_rpc_t ref;

    int err = nvm_rpc_bind_local(&ref, manager);
    if (err != 0)
    {
        throw std::runtime_error("Failed to create reference: " + std::string(std::strerror(err)));
    }

    void* info_ptr = DMA_VADDR(queue_map->vaddr, queue_map->page_size, 2);
    uint64_t info_addr = queue_map->ioaddrs[2];

    nvm_ns_info_t info;
    err = nvm_rpc_ns_info(&info, ref, ns_id, info_ptr, info_addr);
    if (err != 0)
    {
        nvm_rpc_unbind(ref);
        throw std::runtime_error("Failed to get namespace information: " + std::string(std::strerror(err)));
    }

    nvm_rpc_unbind(ref);

    return info.lba_data_size;
}

