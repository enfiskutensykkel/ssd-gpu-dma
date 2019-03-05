#ifndef __linux__
#error "Must compile for Linux"
#endif

#ifdef _CUDA
#ifndef __CUDA__
#define __CUDA__
#endif
#endif

#include <nvm_types.h>
#include <nvm_util.h>
#include <nvm_dma.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include "util.h"
#include "ctrl.h"
#include "dma.h"
#include "linux/map.h"
#include "dprintf.h"



static void remove_mapping_descriptor(struct ioctl_mapping* md)
{
    if (md->type == MAP_TYPE_API)
    {
        free((void*) md->buffer);
    }

    free(md);
}



static void release_mapping_descriptor(struct va_range* va)
{
    remove_mapping_descriptor(_nvm_container_of(va, struct ioctl_mapping, range));
}



static int create_mapping_descriptor(struct ioctl_mapping** handle, enum mapping_type type, void* buffer, size_t size)
{
    size_t page_size = 0;

    switch (type)
    {
        case MAP_TYPE_API:
        case MAP_TYPE_HOST:
            page_size = _nvm_host_page_size();
            break;

#ifdef _CUDA
        case MAP_TYPE_CUDA:
            page_size = (1ULL << 16);
            break;
#endif

        default:
            dprintf("Unknown memory type\n");
            return EINVAL;
    }

    size_t n_pages = NVM_PAGE_ALIGN(size, page_size) / page_size;

    struct ioctl_mapping* md = malloc(sizeof(struct ioctl_mapping));
    if (md == NULL)
    {
        dprintf("Failed to allocate mapping descriptor: %s\n", strerror(errno));
        return errno;
    }

    md->type = type;
    md->buffer = buffer;
    md->range.vaddr = (volatile void*) buffer;
    md->range.page_size = page_size;
    md->range.n_pages = n_pages;

    *handle = md;
    return 0;
}



int nvm_dma_create(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, size_t size)
{
    void* buffer;
    struct ioctl_mapping* md;
    size = NVM_CTRL_ALIGN(ctrl, size);

    *handle = NULL;
    if (_nvm_ctrl_type(ctrl) != DEVICE_TYPE_IOCTL)
    {
        return EBADF;
    }

    int err = posix_memalign(&buffer, ctrl->page_size, size);
    if (err != 0)
    {
        dprintf("Failed to allocate page-aligned memory buffer: %s\n", strerror(err));
        return err;
    }

    err = create_mapping_descriptor(&md, MAP_TYPE_API, buffer, size);
    if (err != 0)
    {
        free(buffer);
        return err;
    }

    err = _nvm_dma_init(handle, ctrl, &md->range, false, &release_mapping_descriptor);
    if (err != 0)
    {
        remove_mapping_descriptor(md);
        return err;
    }

    return 0;
}



int nvm_dma_map_host(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, void* vaddr, size_t size)
{
    struct ioctl_mapping* md;
    *handle = NULL;

    if (_nvm_ctrl_type(ctrl) != DEVICE_TYPE_IOCTL)
    {
        return EBADF;
    }

    int err = create_mapping_descriptor(&md, MAP_TYPE_HOST, vaddr, size);
    if (err != 0)
    {
        return err;
    }

    err = _nvm_dma_init(handle, ctrl, &md->range, false, &release_mapping_descriptor);
    if (err != 0)
    {
        remove_mapping_descriptor(md);
        return err;
    }

    return 0;
}



#ifdef _CUDA
int nvm_dma_map_device(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, void* devptr, size_t size)
{
    struct ioctl_mapping* md;
    *handle = NULL;

    if (_nvm_ctrl_type(ctrl) != DEVICE_TYPE_IOCTL)
    {
        return EBADF;
    }

    int err = create_mapping_descriptor(&md, MAP_TYPE_CUDA, devptr, size);
    if (err != 0)
    {
        return err;
    }

    err = _nvm_dma_init(handle, ctrl, &md->range, false, &release_mapping_descriptor);
    if (err != 0)
    {
        remove_mapping_descriptor(md);
        return err;
    }

    return 0;
}
#endif
