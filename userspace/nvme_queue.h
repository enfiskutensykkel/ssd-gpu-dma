#ifndef __NVME_QUEUE_H__
#define __NVME_QUEUE_H__
#ifdef __cplusplus
extern "C" {
#endif

#include "nvme.h"


int create_queue_pair(nvm_controller_t controller, volatile void* register_ptr);


#ifdef __cplusplus
}
#endif
#endif
