#ifndef __NVME_CTRL_H__
#define __NVME_CTRL_H__
#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"
#include <stddef.h>


/*
 * Reset the actual NVMe controller and initialise the controller structure.
 *
 * ctrl             pointer to controller handle
 *
 * ioctl_fd         open file descriptor to kernel module
 *                  used to call kernel functions for page-locking and aquiring
 *                  physical address of queues
 *
 * register_ptr     mmap'd BAR0 address space of the controller
 *
 * db_size          maximum space used to map in doorbell registers
 *
 * Returns 0 on success, or an error code on failure.
 *
 */
int nvm_init(nvm_controller_t* ctrl, int ioctl_fd, volatile void* register_ptr, size_t db_size);


/*
 * Free allocated resources.
 */
void nvm_free(nvm_controller_t ctrl, int ioctl_fd);


/*
 * Allocate and prepare queue handles
 *
 * Allocate queue handles and set initial queue state.
 * Note that this does not actually allocate the memory used by the queue,
 * the caller must make the necessary calls to create memory handles for the
 * queues.
 *
 * Returns 0 and sets cq and sq on success, or an errno indicating error on failure.
 */
int nvm_prepare_queues(nvm_controller_t ctrl, nvm_queue_t* cq_handle, nvm_queue_t* sq_handle);


/* 
 * Submit prepared queues
 *
 * This will send the necessary NVM commands to the controller and set up the
 * prepared queues. The caller must be sure that all queues are prepared and
 * have associated memory handles before calling this.
 *
 * This function must only be called once.
 *
 * Returns 0 on success or an errno indicating the error on failure.
 */
int nvm_commit_queues(nvm_controller_t ctrl);



#ifdef __cplusplus
}
#endif
#endif
