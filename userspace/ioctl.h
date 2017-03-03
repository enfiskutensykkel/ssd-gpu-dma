#ifndef __IOCTL_H__
#define __IOCTL_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>


int lock_mem(int fd, void* vaddr, size_t size);


int lock_dev_mem(int fd, int dev, void* addr, size_t size);


uint64_t get_phys_addr(int fd, int handle);


int unlock_mem(int fd, int handle);


#ifdef __cplusplus
}
#endif
#endif
