#ifndef __MEMORY_GPU_H__
#define __MEMORY_GPU_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include "types.h"


/* Allocate pinned device memory
 *
 * Allocate a device memory range, pin it, and return a handle for it.
 *
 * Returns a device memory handle on success, or NULL on failure.
 */
memory_t* get_gpu_buffer(int ioctl_fd, int device, size_t size);


/* Release pinned memory
 *
 * Release pinned device memory described by the handle.
 * After calling this function, the memory handle is invalid.
 */
void put_gpu_buffer(int ioctl_fd, memory_t* mem_handle);


/* Allocate a page of device memory and pin it */
int get_gpu_page(int ioctl_fd, int device, page_t* page);


/* Release an allocated page */
void put_gpu_page(int ioctl_fd, page_t* page);


#ifdef __cplusplus
}
#endif
#endif
