#ifndef _SISCI
#error "Must compile with SISCI support"
#endif

#ifndef __DIS_CLUSTER__
#define __DIS_CLUSTER__
#endif

#include <nvm_types.h>
#include <nvm_util.h>
#include <nvm_dma.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "dma.h"
#include "ctrl.h"
#include "dis/device.h"
#include "dis/map.h"
#include "dprintf.h"
#include <sisci_types.h>
#include <sisci_error.h>
#include <sisci_api.h>



/* 
 * Map local segment into virtual address space.
 */
static int va_map_local(struct va_map* m, struct va_range* va, sci_local_segment_t segment, bool write)
{
    sci_error_t err;
    unsigned int flags = 0;
    void* ptr = NULL;
    size_t size = 0;

    flags = !write ? SCI_FLAG_READONLY_MAP : 0;

    size = SCIGetLocalSegmentSize(segment);
    ptr = SCIMapLocalSegment(segment, &m->md, 0, size, NULL, flags, &err);

    switch (err)
    {
        case SCI_ERR_OK:
            m->mapped = true;
            va->remote = false;
            va->n_pages = 1;
            va->page_size = size;
            va->vaddr = (volatile void*) ptr;
            return 0;

        case SCI_ERR_FLAG_NOT_IMPLEMENTED:
        case SCI_ERR_ILLEGAL_FLAG:
        case SCI_ERR_OUT_OF_RANGE:
        case SCI_ERR_SIZE_ALIGNMENT:
        case SCI_ERR_OFFSET_ALIGNMENT:
            return EINVAL;

        default:
            dprintf("Mapping local segment into virtual address space failed: %s\n", _SCIGetErrorString(err));
            return EIO;
    }
}



/* 
 * Map remote segment into virtual address space.
 */
static int va_map_remote(struct va_map* m, struct va_range* va, sci_remote_segment_t segment, bool write, bool wc)
{
    sci_error_t err;
    unsigned int flags = 0;
    volatile void* ptr = NULL;
    size_t size = 0;

    flags |= !write ? SCI_FLAG_READONLY_MAP : 0;
    flags |= !wc ? SCI_FLAG_IO_MAP_IOSPACE : 0;

    size = SCIGetRemoteSegmentSize(segment);
    ptr = SCIMapRemoteSegment(segment, &m->md, 0, size, NULL, flags, &err);

    switch (err)
    {
        case SCI_ERR_OK:
            m->mapped = true;
            va->remote = true;
            va->n_pages = 1;
            va->page_size = size;
            va->vaddr = ptr;
            return 0;

        case SCI_ERR_FLAG_NOT_IMPLEMENTED:
        case SCI_ERR_ILLEGAL_FLAG:
        case SCI_ERR_OUT_OF_RANGE:
        case SCI_ERR_SIZE_ALIGNMENT:
        case SCI_ERR_OFFSET_ALIGNMENT:
            return EINVAL;

        default:
            dprintf("Mapping local segment into virtual address space failed: %s\n", _SCIGetErrorString(err));
            return EIO;
    }
}



/*
 * Unmap segment.
 */
static void va_unmap(struct va_map* m)
{
    if (m->mapped)
    {
        sci_error_t err;

        do
        {
            SCIUnmapSegment(m->md, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

        m->mapped = false;

#ifndef NDEBUG
        if (err != SCI_ERR_OK)
        {
            dprintf("Failed to unmap segment: %s\n", _SCIGetErrorString(err));
        }
#endif
    }
}



/*
 * Helper function to create local segment descriptor.
 */
static int create_local_desc(struct local_segment** ls, struct controller* ctrl, uint32_t adapter, sci_local_segment_t segment)
{
    struct local_segment* s;

    s = (struct local_segment*) malloc(sizeof(struct local_segment));
    if (s == NULL)
    {
        dprintf("Failed to allocate local segment descriptor: %s\n", strerror(errno));
        return errno;
    }

    s->ctrl = ctrl;
    s->adapter = adapter;
    s->segment = segment;
    s->remove = false;
    s->map.mapped = false;
    s->map.md = NULL;
    s->range = VA_RANGE_INIT(false, NULL, SCIGetLocalSegmentSize(s->segment), 1);

    *ls = s;
    return 0;
}



/*
 * Helper function to remove local segment descriptor
 */
static void remove_local_desc(struct local_segment* ls)
{
    if (ls == NULL)
    {
        return;
    }

    va_unmap(&ls->map);

    // Check if the local segment was created by the API
    if (ls->remove)
    {
        sci_error_t err;

        do
        {
            SCISetSegmentUnavailable(ls->segment, ls->adapter, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

#ifndef NDEBUG
        if (err != SCI_ERR_OK)
        {
            dprintf("Failed to set local segment unavailable on adapter %u: %s\n",
                    ls->adapter, _SCIGetErrorString(err));
        }
#endif

        _nvm_local_memory_put(&ls->segment);
    }

    _nvm_ctrl_put(ls->ctrl);
    free(ls);
}



/*
 * Helper function to create remote segment descriptor
 */
static int create_remote_desc(struct remote_segment** rs, struct controller* ctrl, sci_remote_segment_t segment)
{
    struct remote_segment* s;

    s = (struct remote_segment*) malloc(sizeof(struct remote_segment));
    if (s == NULL)
    {
        dprintf("Failed to allocate remote segment descriptor: %s\n", strerror(errno));
        return errno;
    }

    s->ctrl = ctrl;
    s->segment = segment;
    s->disconnect = false;
    s->map.mapped = false;
    s->map.md = NULL;
    s->range = VA_RANGE_INIT(true, NULL, SCIGetRemoteSegmentSize(s->segment), 1);

    *rs = s;

    return 0;
}



/*
 * Helper function to remove remote segment descriptor.
 */
static void remove_remote_desc(struct remote_segment* rs)
{
    if (rs == NULL)
    {
        return;
    }

    va_unmap(&rs->map);

    // Check if segment is connected by the API
    if (rs->disconnect)
    {
        _nvm_device_memory_put(&rs->segment);
    }

    _nvm_ctrl_put(rs->ctrl);
    free(rs);
}



/*
 * Release segment mapping descriptor.
 * Dispatch the correct release function based on type (remote vs. local).
 */
static void release_range(struct va_range* va)
{
    if (va == NULL)
    {
        return;
    }

    if (va->remote)
    {
        remove_remote_desc(_nvm_container_of(va, struct remote_segment, range));
    }
    else
    {
        remove_local_desc(_nvm_container_of(va, struct local_segment, range));
    }
}



/*
 * Create DMA mapping for a local segment.
 */
int nvm_dis_dma_map_local(nvm_dma_t** map, const nvm_ctrl_t* ctrl, uint32_t adapter, sci_local_segment_t lseg, bool map_va)
{
    int status;
    struct local_segment* ls;
    struct controller* ref;

    *map = NULL;

    ref = _nvm_ctrl_get(ctrl);
    if (ref == NULL)
    {
        return ENOSYS;
    }

    status = create_local_desc(&ls, ref, adapter, lseg);
    if (status != 0)
    {
        _nvm_ctrl_put(ref);
        return status;
    }

    if (map_va)
    {
        status = va_map_local(&ls->map, &ls->range, ls->segment, true);
        if (status != 0)
        {
            remove_local_desc(ls);
            return status;
        }
    }

    status = _nvm_dma_init(map, ctrl, &ls->range, &release_range);
    if (status != 0)
    {
        remove_local_desc(ls);
        return status;
    }

    return 0;
}



/*
 * Create DMA mapping for a remote segment.
 */
int nvm_dis_dma_map_remote(nvm_dma_t** map, const nvm_ctrl_t* ctrl, sci_remote_segment_t rseg, bool map_va, bool map_wc)
{
    int status;
    struct remote_segment* rs;
    struct controller* ref;

    *map = NULL;

    if (!map_va && map_wc)
    {
        return EINVAL;
    }

    ref = _nvm_ctrl_get(ctrl);
    if (ref == NULL)
    {
        return ENOSYS;
    }

    status = create_remote_desc(&rs, ref, rseg);
    if (status != 0)
    {
        _nvm_ctrl_put(ref);
        return status;
    }

    if (map_va)
    {
        status = va_map_remote(&rs->map, &rs->range, rs->segment, true, map_wc);
        if (status != 0)
        {
            remove_remote_desc(rs);
            return status;
        }
    }

    status = _nvm_dma_init(map, ctrl, &rs->range, &release_range);
    if (status != 0)
    {
        remove_remote_desc(rs);
        return status;
    }

    return 0;
}



/*
 * Helper function to create a local segment and map it.
 */
static int create_local_segment(struct va_range** va, const nvm_ctrl_t* ctrl, size_t size, void* ptr, bool gpu)
{
    int status;
    struct controller* ref;
    struct local_segment* ls;
    sci_local_segment_t lseg;
    uint32_t adapter;

    if (size == 0)
    {
        return ERANGE;
    }

    // Take controller reference
    ref = _nvm_ctrl_get(ctrl);
    if (ref == NULL)
    {
        return ENOSYS;
    }

    // Create SISCI segment
    status = _nvm_local_memory_get(&lseg, &adapter, ref->device, size, ptr, gpu);
    if (status != 0)
    {
        _nvm_ctrl_put(ref);
        return status;
    }

    // Create local segment descriptor
    status = create_local_desc(&ls, ref, adapter, lseg);
    if (status != 0)
    {
        _nvm_local_memory_put(&lseg);
        _nvm_ctrl_put(ref);
        return status;
    }

    // Indicate that we need cleaning up
    ls->remove = true;
    
    // Map local segment into virtual address space unless it's physical memory
    if (ptr == NULL)
    {
        status = va_map_local(&ls->map, &ls->range, ls->segment, true);
        if (status != 0)
        {
            remove_local_desc(ls); // this also removes segment and puts reference
            return status;
        }
    }
    else
    {
        // XXX Quick and dirty hack
        ls->range.vaddr = (volatile void*) ptr;
    }

    *va = &ls->range;
    return 0;
}



/*
 * Helper function to create a device memory segment, connect to it and map it.
 */
static int create_remote_segment(struct va_range** va, const nvm_ctrl_t* ctrl, size_t size, unsigned int hints)
{
    int status;
    uint32_t id;
    struct controller* ref;
    struct remote_segment* rs;
    sci_remote_segment_t rseg;
    sci_error_t err;

    // Take controller reference
    ref = _nvm_ctrl_get(ctrl);
    if (ref == NULL)
    {
        return ENOSYS;
    }

    // Generate "unique" segment identifier
    status = _nvm_mutex_lock(&ref->device->lock);
    if (status != 0)
    {
        _nvm_ctrl_put(ref);
        return status;
    }

    id = ++ref->device->counter;
    _nvm_mutex_unlock(&ref->device->lock);

    // Attempt to create device memory segment
    SCICreateDeviceSegment(ref->device->device, id, size, SCI_MEMTYPE_PRIVATE, hints, 0, &err);
    if (err != SCI_ERR_OK)
    {
        _nvm_ctrl_put(ref);
        dprintf("Failed to create device segment: %s\n", _SCIGetErrorString(err));
        return ENOSPC;
    }

    // Connect to the device segment
    status = _nvm_device_memory_get(&rseg, ref->device, id, SCI_MEMTYPE_PRIVATE);
    if (status != 0)
    {
        _nvm_ctrl_put(ref);
        return status;
    }

    // Create remote segment descriptor
    status = create_remote_desc(&rs, ref, rseg);
    if (status != 0)
    {
        _nvm_device_memory_put(&rseg);
        _nvm_ctrl_put(ref);
        return status;
    }

    // Indicate that we need some cleaning up
    rs->disconnect = true;

    // Map remote segment into local address space
    status = va_map_remote(&rs->map, &rs->range, rs->segment, true, true);
    if (status != 0)
    {
        remove_remote_desc(rs);
        return status;
    }

    *va = &rs->range;
    return 0;
}



/*
 * Create segment and map it.
 */
int nvm_dis_dma_create(nvm_dma_t** map, const nvm_ctrl_t* ctrl, size_t size, unsigned int mem_hints)
{
    int status;
    struct va_range* va = NULL;

    *map = NULL;

    size = NVM_CTRL_ALIGN(ctrl, size);
    if (size == 0)
    {
        return EINVAL;
    }

    if (mem_hints == 0)
    {
        status = create_local_segment(&va, ctrl, size, NULL, false);
    }
    else
    {
        status = create_remote_segment(&va, ctrl, size, mem_hints);
    }

    if (status != 0)
    {
        return status;
    }

    status = _nvm_dma_init(map, ctrl, va, &release_range);
    if (status != 0)
    {
        release_range(va);
        return status;
    }

    return 0;
}



int nvm_dis_dma_map_host(nvm_dma_t** map, const nvm_ctrl_t* ctrl, void* vaddr, size_t size)
{
    int status;
    struct va_range* va = NULL;

    *map = NULL;

    if (vaddr == NULL)
    {
        return EINVAL;
    }

    size = NVM_CTRL_ALIGN(ctrl, size);

    status = create_local_segment(&va, ctrl, size, vaddr, false);
    if (status != 0)
    {
        return status;
    }

    status = _nvm_dma_init(map, ctrl, va, &release_range);
    if (status != 0)
    {
        release_range(va);
        return status;
    }

    return 0;
}



#ifdef _CUDA
/*
 * Map CUDA device memory for disk controller.
 */
int nvm_dis_dma_map_device(nvm_dma_t** map, const nvm_ctrl_t* ctrl, void* devptr, size_t size)
{
    int status;
    struct va_range* va = NULL;

    *map = NULL;

    if (devptr == NULL)
    {
        return EINVAL;
    }

    size = NVM_PAGE_ALIGN(size, (1ULL << 16));

    status = create_local_segment(&va, ctrl, size, devptr, true);
    if (status != 0)
    {
        return status;
    }

    status = _nvm_dma_init(map, ctrl, va, &release_range);
    if (status != 0)
    {
        release_range(va);
        return status;
    }

    return 0;
}
#endif



uint32_t nvm_dis_node_from_dma(const nvm_dma_t* handle)
{
    if (handle != NULL)
    {
        const struct va_range* va;
        const nvm_ctrl_t* ctrl;

        ctrl = nvm_ctrl_from_dma(handle);
        if (ctrl == NULL || _nvm_ctrl_type(ctrl) != DEVICE_TYPE_SMARTIO)
        {
            return 0;
        }
       
        va = _nvm_dma_va(handle);
        if (va == NULL || va->n_pages != 1)
        {
            return 0;
        }

        if (va->remote)
        {
            const struct remote_segment* rseg = _nvm_container_of(va, struct remote_segment, range);

            return SCIGetRemoteSegmentNodeId(rseg->segment);
        }
        else
        {
            sci_error_t err;
            uint32_t node_id = 0;
            const struct local_segment* lseg = _nvm_container_of(va, struct local_segment, range);

            SCIGetLocalNodeId(lseg->adapter, &node_id, 0, &err);
            if (err == SCI_ERR_OK)
            {
                return node_id;
            }
        }
    }

    return 0;
}

