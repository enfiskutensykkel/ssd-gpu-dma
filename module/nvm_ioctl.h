#ifndef __DIS_NVM_MODULE_IOCTL_H__
#define __DIS_NVM_MODULE_IOCTL_H__

#include <linux/types.h>
#include <asm/ioctl.h>


#define NVM_IOCTL_TYPE          0x80


/* Memory map request */
struct nvm_ioctl_map
{
    uint64_t    vaddr_start;
    size_t      n_pages;
    uint64_t*   ioaddrs;
};



/* Supported operations */
enum nvm_ioctl_type
{
    NVM_MAP_HOST_MEMORY         = _IOW(NVM_IOCTL_TYPE, 1, struct nvm_ioctl_map),
#ifdef __CUDA__
    NVM_MAP_DEVICE_MEMORY       = _IOW(NVM_IOCTL_TYPE, 2, struct nvm_ioctl_map),
#endif
    NVM_UNMAP_MEMORY            = _IOW(NVM_IOCTL_TYPE, 3, uint64_t)
};




#ifndef __KERNEL__
#undef __user
#endif
#endif /* __DIS_NVM_MODULE_IOCTL_H__ */
