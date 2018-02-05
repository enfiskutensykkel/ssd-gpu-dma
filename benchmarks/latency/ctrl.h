#ifndef __CTRL_H__
#define __CTRL_H__

#include <nvm_types.h>
#include <cstdint>
#include "buffer.h"


struct Controller
{
    nvm_ctrl_t*             ctrl;
    nvm_aq_ref              aq_ref;
    int                     fd;
    BufferPtr               aq_mem;
    struct nvm_ctrl_info    info;
    struct nvm_ns_info      ns;
    uint16_t                numQueues;

#ifdef __DIS_CLUSTER__
    Controller(uint64_t controllerId, uint32_t adapter, uint32_t segmentId, uint32_t nvmNamespace, uint16_t numQueues);
#else
    Controller(const char* path, uint32_t nvmNamespace, uint16_t numQueues);
#endif
    ~Controller();
};


#endif
