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
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <fcntl.h>
#include "ctrl.h"
#include "dma.h"
#include "ioctl.h"
#include "util.h"
#include "regs.h"
#include "dprintf.h"



/*
 * DMA handle container.
 * 
 * Note that this structure is of variable size due to the list of pages
 * at the end of the DMA handle.
 */
struct __attribute__((aligned (64))) dma
{
    struct dma_map*         map;        // DMA mapping descriptor
    dma_map_free_t          release;    // Free mapping descriptor
    nvm_dma_t               handle;     // DMA mapping handle
};





/*
 * Memory mapping using kernel module.
 * Indicates type of memory we have mapped.
 */
enum map_type
{
    _MAP_TYPE_HOST      = 0x01,
    _MAP_TYPE_CUDA      = 0x02,
    _MAP_TYPE_INTERNAL  = 0x03
};



/*
 * Memory mapping descriptor.
 * Describes memory mapped using the kernel module.
 */
struct ioctl_mapping
{
    struct dma_map      mapping;        // DMA mapping descriptor
    enum map_type       type;           // Type of memory
    void*               buffer;         // Buffer pointer
    int                 ioctl_fd;       // File descriptor to kernel module
    bool                mapped;         // Indicates if memory is mapped
};



/* Get handle container */
#define container(m) \
    ((struct dma*) (((unsigned char*) (m)) - offsetof(struct dma, handle)))



/* Calculate number of controller pages */
#define n_ctrl_pages(ctrl, page_size, n_pages) \
    (((page_size) * (n_pages)) / (ctrl)->page_size)



/* 
 * Initialize the mapping handle.
 * Sets handle members and populates page address list.
 */
void _nvm_dma_handle_populate(nvm_dma_t* handle, const nvm_ctrl_t* ctrl, const uint64_t* ioaddrs)
{
    const struct dma_map* md = container(handle)->map;

    size_t i_page;
    size_t page_size = md->page_size;
    size_t ctrl_page_size = ctrl->page_size;

    // Set handle members
    handle->vaddr = md->vaddr;
    handle->page_size = ctrl->page_size;
    handle->n_ioaddrs = n_ctrl_pages(ctrl, page_size, md->n_pages);

    // Calculate logical page addresses
    for (i_page = 0; i_page < handle->n_ioaddrs; ++i_page)
    {
        size_t current_page = (i_page * ctrl_page_size) / page_size;
        size_t offset_within_page = (i_page * ctrl_page_size) % page_size;

        handle->ioaddrs[i_page] = ioaddrs[current_page] + offset_within_page;
    }
}



/*
 * Create a DMA handle container.
 *
 * Need to use the controller reference to calculate from local page size 
 * to controller page size.
 */
int _nvm_dma_create(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, struct dma_map* md, dma_map_free_t release)
{
    *handle = NULL;

    size_t page_size = md->page_size;
    size_t n_pages = md->n_pages;

    // Do some sanity checking
    if (page_size == 0 || n_pages == 0 || (page_size * n_pages) % ctrl->page_size != 0)
    {
        dprintf("Addresses do not align with controller pages");
        return EINVAL;
    }

    // Size of the handle container
    size_t container_size = sizeof(struct dma) + (n_ctrl_pages(ctrl, page_size, n_pages)) * sizeof(uint64_t);

    // Allocate the container and set mapping
    struct dma* container = (struct dma*) malloc(container_size);
    if (container == NULL)
    {
        dprintf("Failed to allocate DMA descriptor: %s\n", strerror(errno));
        return ENOMEM;
    }

    container->map = md;
    container->release = release;
    container->handle.vaddr = NULL;
    container->handle.page_size = 0;
    container->handle.n_ioaddrs = 0;

    *handle = &container->handle;
    return 0;
}



/*
 * Free DMA handle.
 */
void _nvm_dma_remove(nvm_dma_t* handle)
{
    struct dma* dma = container(handle);

    if (dma->release != NULL)
    {
        dma->release(dma->map);
    }
    free(dma);
}



/*
 * Create DMA mapping descriptor from physical/bus addresses.
 */
int nvm_dma_map(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, void* vaddr, size_t page_size, size_t n_pages, const uint64_t* ioaddrs)
{
    int status;
    *handle = NULL;

    struct dma_map* map = malloc(sizeof(struct dma_map));
    if (map == NULL)
    {
        dprintf("Failed to allocate mapping descriptor: %s\n", strerror(errno));
        return ENOMEM;
    }

    map->vaddr = vaddr;
    map->page_size = page_size;
    map->n_pages = n_pages;

    status = _nvm_dma_create(handle, ctrl, map, (dma_map_free_t) free);
    if (status != 0)
    {
        free(map);
        return status;
    }

    _nvm_dma_handle_populate(*handle, ctrl, ioaddrs);
    return 0;
}



/*
 * Remove DMA mapping descriptor.
 */
void nvm_dma_unmap(nvm_dma_t* handle)
{
    if (handle != NULL)
    {
        _nvm_dma_remove(handle);
    }
}





/*
 * Helper function to lock pages and retrieve IO addresses for a 
 * virtual memory range.
 */
static int map_memory(struct ioctl_mapping* md, uint64_t* ioaddrs)
{
    enum nvm_ioctl_type type;
    
    switch (md->type)
    {
        case _MAP_TYPE_INTERNAL:
        case _MAP_TYPE_HOST:
            type = NVM_MAP_HOST_MEMORY;
            break;

#ifdef _CUDA
        case _MAP_TYPE_CUDA:
            type = NVM_MAP_DEVICE_MEMORY;
            break;
#endif

        default:
            dprintf("Unknown memory type\n");
            return EINVAL;
    }

    struct nvm_ioctl_map request = {
        .vaddr_start = (uintptr_t) md->mapping.vaddr,
        .n_pages = md->mapping.n_pages,
        .ioaddrs = ioaddrs
    };

    int err = ioctl(md->ioctl_fd, type, &request);
    if (err < 0)
    {
        dprintf("Page mapping kernel request failed: %s\n", strerror(errno));
        return errno;
    }
    
    md->mapped = true;
    return 0;
}



/*
 * Release locked pages.
 */
static int unmap_memory(const struct ioctl_mapping* md)
{
    uint64_t addr = (uintptr_t) md->mapping.vaddr;

    int err = ioctl(md->ioctl_fd, NVM_UNMAP_MEMORY, &addr);

    if (err < 0)
    {
        dprintf("Page unmapping kernel request failed: %s\n", strerror(errno));
        return errno;
    }

    return 0;
}



static int create_mapping(struct ioctl_mapping** handle, enum map_type type, int fd, void* vaddr, size_t size)
{
    size_t page_size = 0;

    switch (type)
    {
        case _MAP_TYPE_HOST:
        case _MAP_TYPE_INTERNAL:
            page_size = _nvm_host_page_size();
            break;

#ifdef _CUDA
        case _MAP_TYPE_CUDA:
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
        return ENOMEM;
    }

    md->mapping.vaddr = vaddr;
    md->mapping.page_size = page_size;
    md->mapping.n_pages = n_pages;
    md->type = type;
    md->mapped = false;
    md->buffer = NULL;

    if (type == _MAP_TYPE_INTERNAL)
    {
        md->buffer = vaddr;
    }

    md->ioctl_fd = dup(fd);
    if (md->ioctl_fd < 0)
    {
        dprintf("Failed to duplicate file descriptor: %s\n", strerror(errno));
        free(md);
        return EBADF;
    }

    *handle = md;
    return 0;
}



/*
 * Remove mapping descriptor.
 */
static void remove_mapping(struct ioctl_mapping* md)
{
    if (md->mapped)
    {
        unmap_memory(md);
    }
    close(md->ioctl_fd);
    free(md->buffer);
    free(md);
}



/*
 * Helper function to map an address range and initialize DMA handle.
 */
static int populate_handle(struct dma* container, const nvm_ctrl_t* ctrl)
{
    uint64_t* ioaddrs = calloc(container->map->n_pages, sizeof(uint64_t));
    if (ioaddrs == NULL)
    {
        return ENOMEM;
    }

    int err = map_memory((struct ioctl_mapping*) container->map, ioaddrs);
    if (err != 0)
    {
        free(ioaddrs);
        return err;
    }

    _nvm_dma_handle_populate(&container->handle, ctrl, ioaddrs);
    free(ioaddrs);
    
    return 0;
}




#if defined ( __unix__ ) 
int nvm_dma_create(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, size_t size)
{
    void* buffer;
    struct ioctl_mapping* md;
    size = NVM_CTRL_ALIGN(ctrl, size);

    *handle = NULL;
    int fd = _nvm_fd_from_ctrl(ctrl);
    if (fd < 0)
    {
        return EBADF;
    }

    int err = posix_memalign(&buffer, ctrl->page_size, size);
    if (err != 0)
    {
        dprintf("Faile to allocate page-aligned buffer: %s\n", strerror(err));
        return err;
    }

    err = create_mapping(&md, _MAP_TYPE_INTERNAL, fd, buffer, size);
    if (err != 0)
    {
        free(buffer);
        return err;
    }
    
    err = _nvm_dma_create(handle, ctrl, (struct dma_map*) md, (dma_map_free_t) remove_mapping);
    if (err != 0)
    {
        remove_mapping(md);
        return err;
    }

    err = populate_handle(container(*handle), ctrl);
    if (err != 0)
    {
        remove_mapping(md);
        *handle = NULL;
        return err;
    }

    return 0;
}
#endif



/*
 * Create DMA mapping descriptor from virtual address using kernel module.
 */
int nvm_dma_map_host(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, void* vaddr, size_t size)
{
    struct ioctl_mapping* md;

    *handle = NULL;

    int fd = _nvm_fd_from_ctrl(ctrl);
    if (fd < 0)
    {
        return EBADF;
    }

    int err = create_mapping(&md, _MAP_TYPE_HOST, fd, vaddr, size);
    if (err != 0)
    {
        return err;
    }
    
    err = _nvm_dma_create(handle, ctrl, (struct dma_map*) md, (dma_map_free_t) remove_mapping);
    if (err != 0)
    {
        remove_mapping(md);
        return err;
    }

    err = populate_handle(container(*handle), ctrl);
    if (err != 0)
    {
        remove_mapping(md);
        *handle = NULL;
        return err;
    }

    return 0;
}




#ifdef _CUDA
int nvm_dma_map_device(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, void* devptr, size_t size)
{
    struct ioctl_mapping* md;

    *handle = NULL;

    int fd = _nvm_fd_from_ctrl(ctrl);
    if (fd < 0)
    {
        return EBADF;
    }

    int err = create_mapping(&md, _MAP_TYPE_CUDA, fd, devptr, size);
    if (err != 0)
    {
        return err;
    }

    err = _nvm_dma_create(handle, ctrl, (struct dma_map*) md, (dma_map_free_t) remove_mapping);
    if (err != 0)
    {
        remove_mapping(md);
        return err;
    }

    err = populate_handle(container(*handle), ctrl);
    if (err != 0)
    {
        remove_mapping(md);
        *handle = NULL;
        return err;
    }

    return 0;
}
#endif

