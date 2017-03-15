#ifndef __NVME_QUEUE_H__
#define __NVME_QUEUE_H__
#ifdef __cplusplus
extern "C" {
#endif

#include "nvme.h"


/*
 * Allocate and prepare queue handles and initialize queue state
 */
int prepare_queue_handles(nvm_controller_t controller);


/* 
 * Submit the prepared queues.
 * This function must only be called once.
 */
int create_queues(nvm_controller_t controller);


#ifdef __cplusplus
}
#endif
#endif
