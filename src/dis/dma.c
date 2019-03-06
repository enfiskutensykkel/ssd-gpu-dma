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
 * Device memory segments are a special case of mappings,
 * where a controller reference must be taken in order to
 * get the segment and map it.
 */
struct segment
{
    struct controller*  ctrl;
    struct map          map;
};



/* 
 * Map segment into virtual address space.
 */
static int va_map(struct map* m, bool write, bool wc)
{
    sci_error_t err;
    unsigned int flags = 0;
    volatile void* ptr = NULL;
    size_t size = 0;

    flags |= !write ? SCI_FLAG_READONLY_MAP : 0;
    flags |= !wc ? SCI_FLAG_IO_MAP_IOSPACE : 0;

    switch (m->type)
    {
        case SEGMENT_TYPE_PHYSICAL:
            return ENOTSUP;

        case SEGMENT_TYPE_LOCAL:
            size = SCIGetLocalSegmentSize(m->lseg);
            ptr = SCIMapLocalSegment(m->lseg, &m->md, 0, size, NULL, 0, &err);
            break;

        case SEGMENT_TYPE_REMOTE:
        case SEGMENT_TYPE_DEVICE:
            size = SCIGetRemoteSegmentSize(m->rseg);
            ptr = SCIMapRemoteSegment(m->rseg, &m->md, 0, size, NULL, 0, &err);
            break;

        default:
            dprintf("Unknown segment type\n");
            return EINVAL;
    }

    switch (err)
    {
        case SCI_ERR_OK:
            m->mapped = true;
            m->range.n_pages = 1;
            m->range.page_size = size;
            m->range.vaddr = ptr;
            return 0;

        case SCI_ERR_FLAG_NOT_IMPLEMENTED:
        case SCI_ERR_ILLEGAL_FLAG:
        case SCI_ERR_OUT_OF_RANGE:
        case SCI_ERR_SIZE_ALIGNMENT:
        case SCI_ERR_OFFSET_ALIGNMENT:
            return EINVAL;

        default:
            dprintf("Mapping segment into virtual address space failed: %s\n", _SCIGetErrorString(err));
            return EIO;
    }
}



/*
 * Unmap segment.
 */
static void va_unmap(struct map* m)
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
 * Release SISCI resources and remove mapping descriptor.
 */
static void remove_map(struct map* m)
{
    va_unmap(m);

    if (m->type == SEGMENT_TYPE_DEVICE)
    {
        struct segment* seg = _nvm_container_of(m, struct segment, map);
        _nvm_disconnect_device_memory(&seg->map.rseg);
        _nvm_ctrl_put(seg->ctrl);
        free(seg);
    }
    else if (m->type == SEGMENT_TYPE_PHYSICAL)
    {
        sci_error_t err;
        struct segment* seg = _nvm_container_of(m, struct segment, map);

        do
        {
            SCIRemoveSegment(seg->map.lseg, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

#ifndef NDEBUG
        if (err != SCI_ERR_OK)
        {
            dprintf("Failed to remove physical segment: %s\n", _SCIGetErrorString(err));
        }
#endif

        _nvm_ctrl_put(seg->ctrl);
        free(seg);
    }
    else
    {
        free(m);
    }
}



static void release_map(struct va_range* va)
{
    struct map* m = _nvm_container_of(va, struct map, range);
    remove_map(m);
}



/*
 * Create mapping descriptor
 */
static int create_map(struct map** md, const nvm_ctrl_t* ctrl, enum segment_type type, size_t size)
{
    struct map* map;

    *md = NULL;

    size = NVM_CTRL_ALIGN(ctrl, size);
    if (size == 0)
    {
        return EINVAL;
    }

    map = (struct map*) malloc(sizeof(struct map));
    if (map == NULL)
    {
        dprintf("Failed to allocate mapping descriptor: %s\n", strerror(errno));
        return errno;
    }

    map->type = type;
    map->lseg = NULL;
    map->rseg = NULL;
    map->adapter = -1;
    map->size = size;
    map->mapped = false;
    map->md = NULL;
    map->range = VA_RANGE_INIT(NULL, map->size, 1);

    *md = map;
    return 0;
}



/*
 * Create mapping descriptor from device memory segment.
 */
static int connect_device_segment(struct map** md, const nvm_ctrl_t* ctrl, uint32_t id, unsigned int memtype)
{
    int err;
    struct segment* s;
    struct map* map;
    sci_remote_segment_t segment;
    struct controller* ref;

    *md = NULL;

    ref = _nvm_ctrl_get(ctrl);
    if (ref == NULL)
    {
        return ENOSPC;
    }

    err = _nvm_connect_device_memory(&segment, ref->device, id, memtype);
    if (err != 0)
    {
        _nvm_ctrl_put(ref);
        return err;
    }

    s = (struct segment*) malloc(sizeof(struct segment));
    if (s == NULL)
    {
        _nvm_disconnect_device_memory(&segment);
        _nvm_ctrl_put(ref);
        dprintf("Failed to allocate segment handle: %s\n", strerror(errno));
        return errno;
    }

    s->ctrl = ref;
    map = &s->map;
    map->type = SEGMENT_TYPE_DEVICE;
    map->lseg = NULL;
    map->rseg = segment;
    map->adapter = -1;
    map->size = NVM_CTRL_ALIGN(ctrl, SCIGetRemoteSegmentSize(segment));
    map->md = NULL;
    map->range = VA_RANGE_INIT(NULL, map->size, 1);

    *md = map;
    return 0;
}



#ifdef _CUDA
static int create_physical_segment(struct map** md, const nvm_ctrl_t* ctrl, uint32_t adapter, void* ptr, size_t size)
{
    sci_error_t err;
    struct segment* s;
    struct map* map;
    sci_local_segment_t segment;
    struct controller* ref;

    *md = NULL;

    ref = _nvm_ctrl_get(ctrl);
    if (ref == NULL)
    {
        return ENOSPC;
    }

    SCICreateSegment(ref->device->sd, &segment, 0, size, NULL, NULL, SCI_FLAG_EMPTY | SCI_FLAG_AUTO_ID, &err);
    if (err != SCI_ERR_OK)
    {
        _nvm_ctrl_put(ref);
        dprintf("Failed to create physical segment: %s\n", _SCIGetErrorString(err));
        return EIO;
    }

    SCIAttachPhysicalMemory(0, ptr, 0, size, segment, SCI_FLAG_CUDA_BUFFER, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to attach physical memory: %s\n", _SCIGetErrorString(err));
        SCIRemoveSegment(segment, 0, &err);
        _nvm_ctrl_put(ref);
        return EIO;
    }

    SCIPrepareSegment(segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to prepare physical segment: %s\n", _SCIGetErrorString(err));
        SCIRemoveSegment(segment, 0, &err);
        _nvm_ctrl_put(ref);
        return EIO;
    }

    s = (struct segment*) malloc(sizeof(struct segment));
    if (s == NULL)
    {
        dprintf("Failed to allocate segment handle: %s\n", strerror(errno));
        SCIRemoveSegment(segment, 0, &err);
        _nvm_ctrl_put(ref);
        return errno;
    }

    s->ctrl = ref;
    map = &s->map;
    map->type = SEGMENT_TYPE_PHYSICAL;
    map->lseg = segment;
    map->rseg = NULL;
    map->adapter = adapter;
    map->size = size;
    map->mapped = false;
    map->md = NULL;
    map->range = VA_RANGE_INIT(NULL, map->size, 1);

    *md = map;
    return 0;
}
#endif



/*
 * Map local for device segment.
 */
int nvm_dis_dma_map_local(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, sci_local_segment_t lseg, uint32_t adapter, bool map_va)
{
    int status;
    struct map* m;

    *handle = NULL;

    status = create_map(&m, ctrl, SEGMENT_TYPE_LOCAL, SCIGetLocalSegmentSize(lseg));
    if (status != 0)
    {
        return status;
    }

    m->adapter = adapter;
    m->lseg = lseg;

    if (map_va)
    {
        status = va_map(m, true, true);
        if (status != 0)
        {
            remove_map(m);
            return status;
        }
    }

    status = _nvm_dma_init(handle, ctrl, &m->range, false, &release_map);
    if (status != 0)
    {
        remove_map(m);
        return status;
    }

    return 0;
}



/*
 * Map remote segment for device.
 */
int nvm_dis_dma_map_remote(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, sci_remote_segment_t rseg, bool map_va, bool map_wc)
{
    int status;
    struct map* m;

    *handle = NULL;

    if (map_wc && !map_va)
    {
        return EINVAL;
    }

    status = create_map(&m, ctrl, SEGMENT_TYPE_REMOTE, SCIGetRemoteSegmentSize(rseg));
    if (status != 0)
    {
        return status;
    }

    m->rseg = rseg;

    if (map_va)
    {
        status = va_map(m, true, map_wc);
        if (status != 0)
        {
            remove_map(m);
            return status;
        }
    }

    status = _nvm_dma_init(handle, ctrl, &m->range, true, &release_map);
    if (status != 0)
    {
        remove_map(m);
        return status;
    }

    return 0;
}



int nvm_dis_dma_map_host(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, uint32_t adapter, void* vaddr, size_t size)
{
    dprintf("Function not implemented\n");
    return ENOTSUP;
}



#ifdef _CUDA
/*
 * Map CUDA device memory for device.
 */
int nvm_dis_dma_map_device(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, uint32_t adapter, void* dptr, size_t size)
{
    int status;
    struct map* m;
    
    size = NVM_PAGE_ALIGN(size, (1ULL << 16));
    *handle = NULL;

    status = create_physical_segment(&m, ctrl, adapter, dptr, size);
    if (status != 0)
    {
        return status;
    }

    // TODO: va_map() when supported by SISCI
    
    status = _nvm_dma_init(handle, ctrl, &m->range, false, &release_map);
    if (status != 0)
    {
        remove_map(m);
        return status;
    }

    return 0;
}
#endif



/*
 * Connect to device memory.
 */
int nvm_dis_dma_connect(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, uint32_t id, bool shared)
{
    int status;
    struct map* m;
    unsigned int memtype = shared ? SCI_MEMTYPE_SHARED : SCI_MEMTYPE_PRIVATE;

    *handle = NULL;

    status = connect_device_segment(&m, ctrl, id, memtype);
    if (status != 0)
    {
        return status;
    }

    // FIXME: TODO: Figure out if segment is local

    status = va_map(m, true, true);
    if (status != 0)
    {
        remove_map(m);
        return status;
    }

    status = _nvm_dma_init(handle, ctrl, &m->range, true, &release_map);
    if (status != 0)
    {
        remove_map(m);
        return status;
    }

    return 0;
}



/*
 * Create device memory.
 */
int nvm_dis_dma_create(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, uint32_t id, size_t size, bool shared, unsigned int hints)
{
    sci_error_t err;
    unsigned int flags = 0;
    struct controller* ref;
    unsigned int memtype = shared ? SCI_MEMTYPE_SHARED : SCI_MEMTYPE_PRIVATE;

    *handle = NULL;
    size = NVM_CTRL_ALIGN(ctrl, size);

    if (hints == 0)
    {
        // FIXME: This fails in the driver for some reason
        hints = SCI_MEMACCESS_HOST_READ | SCI_MEMACCESS_HOST_WRITE | SCI_MEMACCESS_DEVICE_READ | SCI_MEMACCESS_DEVICE_WRITE;
        //flags = SCI_FLAG_LOCAL_ONLY; 
    }

    ref = _nvm_ctrl_get(ctrl);
    if (ref == NULL)
    {
        return ENOTTY;
    }

    SCICreateDeviceSegment(ref->device->device, id, size, memtype, hints, flags, &err);

    _nvm_ctrl_put(ref);

    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to create device segment: %s\n", _SCIGetErrorString(err));
        return EIO;
    }

    return nvm_dis_dma_connect(handle, ctrl, id, shared);
}

