#ifndef __linux__
#error "Must compile for Linux"
#endif

#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_ctrl.h>
#include <nvm_util.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdio.h>
#include "linux/map.h"
#include "linux/ioctl.h"
#include "ctrl.h"
#include "dprintf.h"



/*
 * Device descriptor
 */
struct device
{
    int fd; /* ioctl file descriptor */
};



/*
 * Unmap controller memory and close file descriptor.
 */
static void release_device(struct device* dev, volatile void* mm_ptr, size_t mm_size)
{
    munmap((void*) mm_ptr, mm_size);
    close(dev->fd);
    free(dev);
}



/*
 * Call kernel module ioctl and map memory for DMA.
 */
static int ioctl_map(const struct device* dev, const struct va_range* va, uint64_t* ioaddrs)
{
    const struct ioctl_mapping* m = _nvm_container_of(va, struct ioctl_mapping, range);
    enum nvm_ioctl_type type;

    switch (m->type)
    {
        case MAP_TYPE_API:
        case MAP_TYPE_HOST:
            type = NVM_MAP_HOST_MEMORY;
            break;

#ifdef _CUDA
        case MAP_TYPE_CUDA:
            type = NVM_MAP_DEVICE_MEMORY;
            break;
#endif
        default:
            dprintf("Unknown memory type in map for device");
            return EINVAL;
    }

    struct nvm_ioctl_map request = {
        .vaddr_start = (uintptr_t) m->buffer,
        .n_pages = va->n_pages,
        .ioaddrs = ioaddrs
    };

    int err = ioctl(dev->fd, type, &request);
    if (err < 0)
    {
        dprintf("Page mapping kernel request failed (ptr=%p, n_pages=%zu): %s\n", 
                m->buffer, va->n_pages, strerror(errno));
        return errno;
    }
    
    return 0;
}



/*
 * Call kernel module ioctl and unmap memory.
 */
static void ioctl_unmap(const struct device* dev, const struct va_range* va)
{
    const struct ioctl_mapping* m = _nvm_container_of(va, struct ioctl_mapping, range);
    uint64_t addr = (uintptr_t) m->buffer;
    
    int err = ioctl(dev->fd, NVM_UNMAP_MEMORY, &addr);

#ifndef NDEBUG
    if (err < 0)
    {
        dprintf("Page unmapping kernel request failed: %s\n", strerror(errno));
    }
#endif
}



int nvm_ctrl_init(nvm_ctrl_t** ctrl, int filedes)
{
    int err;
    struct device* dev;
    const struct device_ops ops = {
        .release_device = &release_device,
        .map_range = &ioctl_map,
        .unmap_range = &ioctl_unmap,
    };

    *ctrl = NULL;
    dev = (struct device*) malloc(sizeof(struct device));
    if (dev == NULL)
    {
        dprintf("Failed to allocate device handle: %s\n", strerror(errno));
        return ENOMEM;
    }

    dev->fd = dup(filedes);
    if (dev->fd < 0)
    {
        free(dev);
        dprintf("Could not duplicate file descriptor: %s\n", strerror(errno));
        return errno;
    }

    err = fcntl(dev->fd, F_SETFD, O_RDWR | O_NONBLOCK);
    if (err == -1)
    {
        close(dev->fd);
        free(dev);
        dprintf("Failed to set file descriptor control: %s\n", strerror(errno));
        return errno;
    }

    const size_t mm_size = NVM_CTRL_MEM_MINSIZE;
    void* mm_ptr = mmap(NULL, mm_size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_FILE, dev->fd, 0);
    if (mm_ptr == NULL)
    {
        close(dev->fd);
        free(dev);
        dprintf("Failed to map device memory: %s\n", strerror(errno));
        return errno;
    }

    err = _nvm_ctrl_init(ctrl, dev, &ops, DEVICE_TYPE_IOCTL, mm_ptr, mm_size);
    if (err != 0)
    {
        release_device(dev, mm_ptr, mm_size);
        return err;
    }

    return 0;
}

