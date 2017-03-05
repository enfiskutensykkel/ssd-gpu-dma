#ifndef __NVME_INIT_H__
#define __NVME_INIT_H__
#ifdef __cplusplus
extern "C" {
#endif

#include "nvme.h"


/*
 * Reset the actual NVMe controller and initialise the controller structure.
 *
 * controller       pointer to controller handle
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
int nvm_init(nvm_controller_t* controller, int ioctl_fd, volatile void* register_ptr, size_t db_size);


/*
 * Free allocated resources.
 */
void nvm_free(nvm_controller_t controller, int ioctl_fd);


#ifdef __cplusplus
}
#endif
#endif
