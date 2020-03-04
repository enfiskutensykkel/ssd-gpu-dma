#ifdef _CUDA
#ifndef __CUDA__
#define __CUDA__
#endif
#endif

#include <nvm_types.h>
#include <nvm_dma.h>
#include <nvm_util.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "util.h"
#include "ctrl.h"
#include "dma.h"
#include "mutex.h"
#include "dprintf.h"


typedef void (*va_unmap_t)(const struct device*, const struct va_range*);

/*
 * Reference counted mapping descriptor
 */
struct map
{
    struct mutex        lock;   // Ensure exclusive access
    uint32_t            count;  // Reference count
    struct controller*  ctrl;   // Device reference
    struct va_range*    va;     // Virtual address range

    va_range_free_t     release;// Callback for releasing address range
    va_unmap_t          unmap;// Callback for unmapping address range
};



/*
 * Internal DMA handle container.
 *
 * Note that this structure is of variable size due to the list of pages
 * at the end of the DMA handle.
 */
struct __attribute__((aligned (32))) container
{
    struct map*         map;    // DMA mapping descriptor
    nvm_dma_t           handle; // User handle
};



/* Calculate number of controller pages */
#define n_ctrl_pages(ctrl, page_size, n_pages) \
    (((page_size) * (n_pages)) / (ctrl)->page_size)



/*
 * Create reference counted mapping descriptor.
 */
static int create_map(struct map** md, const nvm_ctrl_t* ctrl, struct va_range* va, va_range_free_t release)
{
    *md = NULL;

#ifndef NDEBUG
    if (release == NULL)
    {
        return EINVAL;
    }
#endif

    // Take device reference
    struct controller* ref = _nvm_ctrl_get(ctrl);
    if (ref == NULL)
    {
        return ENOSPC;
    }

    struct map* m = (struct map*) malloc(sizeof(struct map));
    if (m == NULL)
    {
        _nvm_ctrl_put(ref);
        dprintf("Failed to allocate mapping descriptor: %s\n", strerror(errno));
        return errno;
    }

    int err = _nvm_mutex_init(&m->lock);
    if (err != 0)
    {
        free(m);
        _nvm_ctrl_put(ref);
        return err;
    }

    m->count = 1;
    m->ctrl = ref;
    m->va = va;
    m->release = release;
    m->unmap = NULL;

    *md = m;
    return 0;
}



/*
 * Release mapping descriptor.
 */
static void remove_map(struct map* md)
{
    _nvm_ctrl_put(md->ctrl);
    free(md);
}



/*
 * Helper function to initialize DMA handle members and 
 * populate bus address list.
 */
static void populate_handle(nvm_dma_t* handle, const struct va_range* va, const nvm_ctrl_t* ctrl, const uint64_t* ioaddrs)
{
    size_t i_page;
    size_t page_size = va->page_size;

    // Set handle members
    handle->vaddr = (void*) va->vaddr;
    handle->page_size = ctrl->page_size;
    handle->n_ioaddrs = n_ctrl_pages(ctrl, page_size, va->n_pages);

    // Calculate logical page addresses
    handle->contiguous = true;
    for (i_page = 0; i_page < handle->n_ioaddrs; ++i_page)
    {
        size_t current_page = (i_page * handle->page_size) / page_size;
        size_t offset_within_page = (i_page * handle->page_size) % page_size;

        handle->ioaddrs[i_page] = ioaddrs[current_page] + offset_within_page;

        if (i_page > 0 && handle->ioaddrs[i_page - 1] + handle->page_size != handle->ioaddrs[i_page])
        {
            handle->contiguous = false;
        }
    }
}



/*
 * Helper function to map an address range for device and populate DMA handle.
 */
static int dma_map(struct container* container)
{
    struct map* md = container->map;

    // Check if mapping is supported
    if (md->ctrl->ops.map_range == NULL)
    {
        return EINVAL;
    }

    // Allocate list of bus addresses
    uint64_t* ioaddrs = (uint64_t*) calloc(md->va->n_pages, sizeof(uint64_t));
    if (ioaddrs == NULL)
    {
        return ENOMEM;
    }

    int err = md->ctrl->ops.map_range(md->ctrl->device, md->va, ioaddrs);
    if (err != 0)
    {
        free(ioaddrs);
        return err;
    }
    md->unmap = md->ctrl->ops.unmap_range;

    populate_handle(&container->handle, md->va, &md->ctrl->handle, ioaddrs);
    free(ioaddrs);

    return 0;
}



/*
 * Decrease mapping descriptor reference count.
 */
static void put_map(struct map* md)
{
    if (md == NULL)
    {
        return;
    }

    _nvm_mutex_lock(&md->lock);
    if (--md->count == 0)
    {
        if (md->unmap != NULL)
        {
            md->unmap(md->ctrl->device, md->va);
        }
        md->release(md->va);
        md->va = NULL;
    }
    _nvm_mutex_unlock(&md->lock);

    if (md->va == NULL)
    {
        remove_map(md);
    }
}



/*
 * Increase mapping descriptor reference count.
 */
static int get_map(struct map* md)
{
    int err;

    if (md == NULL)
    {
        return EINVAL;
    }

    err = _nvm_mutex_lock(&md->lock);
    if (err != 0)
    {
        dprintf("Failed to take map reference lock: %s\n", strerror(err));
        return err;
    }

    ++md->count;
    _nvm_mutex_unlock(&md->lock);

    return 0;
}



/*
 * Create a DMA handle container.
 * This function assumes that the device reference has already been increased.
 */
static int create_container(struct container** container, struct map* md)
{
    *container = NULL;
    const struct controller* ctrl = md->ctrl;
    
    size_t page_size = md->va->page_size;
    size_t n_pages = md->va->n_pages;

    // Do some sanity checking
    if (page_size == 0 || n_pages == 0 || (page_size * n_pages) % ctrl->handle.page_size != 0)
    {
        dprintf("Addresses do not align with controller pages\n");
        return EINVAL;
    }

    // Size of the handle container
    size_t container_size = sizeof(struct container) + (n_ctrl_pages(&ctrl->handle, page_size, n_pages)) * sizeof(uint64_t);

    // Allocate the container and set mapping
    *container = (struct container*) malloc(container_size);
    if (*container == NULL)
    {
        dprintf("Failed to allocate DMA descriptor: %s\n", strerror(errno));
        return ENOMEM;
    }

    (*container)->map = md;
    (*container)->handle.contiguous = true;
    (*container)->handle.local = !md->va->remote;

    return 0;
}



/*
 * Call release callback and free container.
 * NB! unmap (if map was successful) must be called
 *     before calling this function.
 */
static void remove_container(struct container* container)
{
    put_map(container->map);
    container->map = NULL;
    free(container);
}



/*
 * Create and initialize a DMA handle, and map a virtual address range 
 * for the controller.
 */
int _nvm_dma_init(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, struct va_range* va, va_range_free_t release)
{
    *handle = NULL;
    struct map* map;
    struct container* container;

    if (release == NULL)
    {
        return EINVAL;
    }

    // Create mapping descriptor
    int err = create_map(&map, ctrl, va, release);
    if (err != 0)
    {
        return err;
    }

    // Create DMA handle container
    err = create_container(&container, map);
    if (err != 0)
    {
        remove_map(map);
        return err;
    }

    // Map controller for device and populate handle
    err = dma_map(container);
    if (err != 0)
    {
        remove_map(map);
        free(container);
        return err;
    }

    *handle = &container->handle;
    return 0;
}



/*
 * Create DMA mapping descriptor from user supplied physical/bus addresses.
 */
int nvm_dma_map(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, void* vaddr, size_t page_size, size_t n_pages, const uint64_t* ioaddrs)
{
    int status;
    struct map* map;
    struct container* container;
    struct va_range* va;

    *handle = NULL;

    if (ioaddrs == NULL)
    {
        return EINVAL;
    }

    // Create virtual address range descriptor
    va = (struct va_range*) malloc(sizeof(struct va_range));
    if (va == NULL)
    {
        dprintf("Failed to allocate mapping descriptor: %s\n", strerror(errno));
        return ENOMEM;
    }

    va->remote = false;
    va->vaddr = (volatile void*) vaddr;
    va->page_size = page_size;
    va->n_pages = n_pages;

    // Create empty mapping descriptor
    status = create_map(&map, ctrl, va, (va_range_free_t) &free);
    if (status != 0)
    {
        free(va);
        return status;
    }
    
    // Create DMA handle container
    status = create_container(&container, map);
    if (status != 0)
    {
        remove_map(map);
        free(va);
        return status;
    }

    populate_handle(&container->handle, va, ctrl, ioaddrs);
    *handle = &container->handle;
    return 0;
}



/*
 * Create DMA mapping descriptor from other descriptor.
 */
int nvm_dma_remap(nvm_dma_t** handle, const nvm_dma_t* other)
{
    int status;
    struct container* container;
    struct map* map = _nvm_container_of(other, struct container, handle)->map;
    struct va_range va;
   
    *handle = NULL;

    // Increase mapping descriptor reference count
    status = get_map(map);
    if (status != 0)
    {
        return status;
    }

    // Create DMA handle container
    status = create_container(&container, map);
    if (status != 0)
    {
        put_map(map); // Will implicitly release device reference
        return status;
    }

    // Hack to get list of bus addresses, since we don't want to
    // actually call map again, simply increase reference count.
    va.remote = !other->local;
    va.vaddr = map->va->vaddr;
    va.page_size = other->page_size;
    va.n_pages = other->n_ioaddrs;
    
    populate_handle(&container->handle, &va, &map->ctrl->handle, other->ioaddrs);
    *handle = &container->handle;

    return 0;
}



/*
 * Remove DMA mapping descriptor.
 */
void nvm_dma_unmap(nvm_dma_t* handle)
{
    if (handle != NULL)
    {
        struct container* dma = _nvm_container_of(handle, struct container, handle);
        remove_container(dma);
    }
}



const nvm_ctrl_t* nvm_ctrl_from_dma(const nvm_dma_t* handle)
{
    if (handle != NULL)
    {
        const struct container* dma;
        dma = _nvm_container_of(handle, struct container, handle);
        return &dma->map->ctrl->handle;
    }

    return NULL;
}



const struct va_range* _nvm_dma_va(const nvm_dma_t* handle)
{
    if (handle != NULL)
    {
        const struct container* dma = _nvm_container_of(handle, struct container, handle);

        if (dma->map != NULL)
        {
            return dma->map->va;
        }
    }

    return NULL;
}

