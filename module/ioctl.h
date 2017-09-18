#ifndef __DIS_NVM_INTERNAL_IOCTL_H__
#define __DIS_NVM_INTERNAL_IOCTL_H__
#ifndef __KERNEL__
#define __user
#endif

#include <asm/ioctl.h>

#define NVM_IOCTL_MAGIC     0x80


/* Supported operations */
enum nvm_ioctl_type
{
    NVM_GET_DEVICE      = _IO('S', NVM_IOCTL_MAGIC | 0x01),
    NVM_PUT_DEVICE      = _IO('S', NVM_IOCTL_MAGIC | 0x02),
    NVM_GET_RAM_PAGES   = _IO('S', NVM_IOCTL_MAGIC | 0x03),
    NVM_GET_GPU_PAGES   = _IO('S', NVM_IOCTL_MAGIC | 0x04),
    NVM_PUT_PAGES       = _IO('S', NVM_IOCTL_MAGIC | 0x00)
};


struct nvm_ioctl_get_device
{
    
};



#ifndef __KERNEL__
#undef __user
#endif
#endif /* __DIS_NVM_INTERNAL_IOCTL_H__ */
