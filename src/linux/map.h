#ifndef __NVM_INTERNAL_LINUX_MAP_H__
#define __NVM_INTERNAL_LINUX_MAP_H__
#ifdef __linux__

#include "linux/ioctl.h"
#include "dma.h"


/*
 * What kind of memory are we mapping.
 */
enum mapping_type
{
    MAP_TYPE_CUDA   =   0x1,   // CUDA device memory
    MAP_TYPE_HOST   =   0x2,   // Host memory (RAM)
    MAP_TYPE_API    =   0x4    // Allocated by the API (RAM)
};



/*
 * Mapping container
 */
struct ioctl_mapping
{
    enum mapping_type   type;   // What kind of memory
    void*               buffer;
    struct va_range     range;  // Memory range descriptor
};


#endif /* __linux__ */
#endif /* __NVM_INTERNAL_LINUX_MAP_H__ */
