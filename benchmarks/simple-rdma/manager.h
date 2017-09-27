#ifndef __SIMPLE_RDMA_MANAGER_H__
#define __SIMPLE_RDMA_MANAGER_H__

#include <memory>
#include <cstddef>
#include <cstdint>
#include <nvm_types.h>


struct Manager;
typedef std::shared_ptr<Manager> ManagerPtr;


struct Manager
{
    nvm_ctrl_t      ctrl;
    nvm_dma_t       queue_map;
    nvm_manager_t   manager;
    nvm_ctrl_info_t ctrl_info;
    
    Manager(uint32_t ctrl_id);
    ~Manager() noexcept;

    size_t get_blk_size(uint32_t ns_id) const;
};


#endif
