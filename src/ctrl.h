#ifndef __NVM_INTERNAL_CTRL_H__
#define __NVM_INTERNAL_CTRL_H__

#include <nvm_types.h>


/* Forward declaration */
struct device;



/*
 * Look up file descriptor from controller handle.
 * Returns a valid file descriptor (positive integer) or -EBADF if the 
 * controller memory is not mapped through this library.
 */
int _nvm_fd_from_ctrl(const nvm_ctrl_t* ctrl);



#ifdef _SISCI
/*
 * Look up device reference from controller handle.
 * Returns a device reference if used, or NULL if not used.
 */
const struct device* _nvm_device_from_ctrl(const nvm_ctrl_t* ctrl);
#endif



#endif /* __NVM_INTERNAL_CTRL_H__ */
