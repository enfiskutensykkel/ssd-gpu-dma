#ifndef __MEMORY_RAM_H__
#define __MEMORY_RAM_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include "types.h"


/* Allocate page-locked memory
 *
 * Allocate a page-locked memory range in RAM and return a handle for it.
 *
 * Returns a memory handle on success, or NULL on failure.
 */
memory_t* get_ram_buffer(int ioctl_fd, size_t size);


/* Release page-locked memory
 *
 * Release page-locked memory described by the handle.
 * After calling this function, the memory handle is invalid.
 */
void put_ram_buffer(int ioctl_fd, memory_t* mem_handle);


/* Allocate a page of memory and page-lock it */
int get_ram_page(int ioctl_fd, page_t* page);


/* Release an allocated page */
void put_ram_page(int ioctl_fd, page_t* page);


#ifdef __cplusplus
}
#endif
#endif
