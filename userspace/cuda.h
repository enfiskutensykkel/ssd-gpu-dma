#ifndef __CUNVME_CUDA_H__
#define __CUNVME_CUDA_H__
#ifdef __cplusplus
extern "C" {
#endif

#include "nvme.h"


int start_kernel(int ioctl_fd, int device, nvm_controller_t controller);


#ifdef __cplusplus
}
#endif
#endif
