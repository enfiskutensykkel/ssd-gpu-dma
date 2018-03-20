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
#include "dis/local.h"
#include "dis/device.h"
#include "dis/map.h"
#include "dprintf.h"
#include <sisci_types.h>
#include <sisci_error.h>
#include <sisci_api.h>



/*
 * SmartIO mapping descriptor.
 * 
 * Holds a device reference and mapping descriptors.
 */
struct map_descriptor
{
    struct dma_map          dma_mapping;    // DMA mapping head
    struct device           device;         // Device reference
    struct va_map           va_mapping;     // Virtual address mapping
    struct io_map           io_mapping;     // Mapping for device
    void*                   segment;        // Reference to segment descriptor (if used)
};



/*
 * Allocate a map descriptor container and take device reference.
 */
static int create_map_descriptor(struct map_descriptor** container, const nvm_ctrl_t* ctrl, size_t segment_size)
{
    // Get device reference from controller
    const struct device* dev = _nvm_device_from_ctrl(ctrl);
    if (dev == NULL)
    {
        dprintf("Controller is not a cluster device\n");
        return EINVAL;
    }

    // Allocate map descriptor container
    struct map_descriptor* map = (struct map_descriptor*) malloc(sizeof(struct map_descriptor));
    if (map == NULL)
    {
        dprintf("Failed to allocate mapping descriptor: %s\n", strerror(errno));
        return ENOMEM;
    }

    // Increase device reference
    int err = _nvm_device_get(&map->device, dev->device_id);
    if (err != 0)
    {
        free(map);
        dprintf("Failed to take device reference: %s\n", strerror(err));
        return err;
    }

    // Clear all members
    VA_MAP_CLEAR(&map->va_mapping);
    IO_MAP_CLEAR(&map->io_mapping);
    map->dma_mapping.vaddr = NULL;
    map->dma_mapping.page_size = segment_size;
    map->dma_mapping.n_pages = 1;
    map->segment = NULL;

    *container = map;
    return 0;
}



/*
 * Unmap mappings, release device reference and free map descriptor.
 */
static void remove_map_descriptor(struct map_descriptor* map)
{
    if (map != NULL)
    {
        _nvm_va_unmap(&map->va_mapping);
        _nvm_io_unmap(&map->io_mapping);
        _nvm_device_put(&map->device);
        free(map);
    }
}



/*
 * Create DMA mapping for a local segment.
 */
int nvm_dis_dma_map_local(nvm_dma_t** handle, 
                          const nvm_ctrl_t* ctrl, 
                          uint32_t adapter, 
                          sci_local_segment_t segment,
                          bool map_va)
{
    struct map_descriptor* md;
    size_t size; 
    *handle = NULL;

    size = NVM_CTRL_ALIGN(ctrl, SCIGetLocalSegmentSize(segment));
    if (size == 0)
    {
        dprintf("Local segment has no size");
        return -ENOSPC;
    }

    // Create mapping descriptor
    int err = create_map_descriptor(&md, ctrl, size);
    if (err != 0)
    {
        return err;
    }

    // Set reverse mapping (device-local mapping)
    err = _nvm_io_map_local(&md->io_mapping, md->device.device, segment, adapter);
    if (err != 0)
    {
        remove_map_descriptor(md);
        return err;
    }

    // Set up virtual address space mapping
    if (map_va)
    {
        err = _nvm_va_map_local(&md->va_mapping, size, segment);
        if (err != 0)
        {
            remove_map_descriptor(md);
            return err;
        }

        md->dma_mapping.vaddr = (void*) md->va_mapping.vaddr;
    }

    // Create handle container
    err = _nvm_dma_create(handle, ctrl, (struct dma_map*) md, (dma_map_free_t) remove_map_descriptor);
    if (err != 0)
    {
        remove_map_descriptor(md);
        return err;
    }

    // Initialize DMA handle
    _nvm_dma_handle_populate(*handle, ctrl, (uint64_t*) &md->io_mapping.ioaddr);

    return 0;
}



/*
 * Create DMA mapping for a remote segment.
 */
int nvm_dis_dma_map_remote(nvm_dma_t** handle, 
                           const nvm_ctrl_t* ctrl, 
                           sci_remote_segment_t segment, 
                           bool map_va, 
                           bool map_wc)
{
    struct map_descriptor* md;
    *handle = NULL;

    // Find segment size
    size_t size = SCIGetRemoteSegmentSize(segment) & NVM_PAGE_MASK(ctrl->page_size);

    // Create mapping descriptor
    int err = create_map_descriptor(&md, ctrl, size);
    if (err != 0)
    {
        return err;
    }

    // Set up device-local mapping
    err = _nvm_io_map_remote(&md->io_mapping, md->device.device, segment);
    if (err != 0)
    {
        remove_map_descriptor(md);
        return err;
    }

    // Map into local address space
    if (map_va)
    {
        err = _nvm_va_map_remote(&md->va_mapping, size, segment, true, map_wc);
        if (err != 0)
        {
            remove_map_descriptor(md);
            return err;
        }

        md->dma_mapping.vaddr = (void*) md->va_mapping.vaddr;
    }

    // Create handle container
    err = _nvm_dma_create(handle, ctrl, (struct dma_map*) md, (dma_map_free_t) remove_map_descriptor);
    if (err != 0)
    {
        remove_map_descriptor(md);
        return err;
    }

    // Initialize DMA handle
    _nvm_dma_handle_populate(*handle, ctrl, (uint64_t*) &md->io_mapping.ioaddr);

    return 0;
}



/*
 * Helper function to connect to a remote segment.
 */
static int connect_remote_segment(struct device_memory** seg, const struct device* dev, uint32_t adapter, uint32_t segno, bool shared)
{
    int status = 0;
    *seg = NULL;

    struct device_memory* s = (struct device_memory*) malloc(sizeof(struct device_memory));
    if (s == NULL)
    {
        dprintf("Failed to allocate segment descriptor: %s\n", strerror(errno));
        return ENOMEM;
    }

    status = _nvm_device_memory_get(s, dev, adapter, segno, shared ? SCI_FLAG_SHARED : SCI_FLAG_PRIVATE);
    if (status != 0)
    {
        free(s);
        return status;
    }

    *seg = s;
    return 0;
}



/*
 * Helper function to allocate a local segment descriptor.
 */
static int create_local_segment(struct local_memory** seg, uint32_t id, size_t size, void* dptr)
{
    int status = 0;
    *seg = NULL;

    struct local_memory* s = (struct local_memory*) malloc(sizeof(struct local_memory));
    if (s == NULL)
    {
        dprintf("Failed to allocate segment descriptor: %s\n", strerror(errno));
        return ENOMEM;
    }

    if (dptr == NULL)
    {
        status = _nvm_local_memory_get(s, id, size);
    }
    else
    {
        status = _nvm_local_memory_get_attached(s, id, dptr, size);
    }

    if (status != 0)
    {
        free(s);
        return status;
    }

    *seg = s;
    return 0;
}



/*
 * Helper function to unmap and remove local segment.
 */
static void remove_local_segment(struct map_descriptor* md)
{
    if (md != NULL)
    {
        if (md->segment != NULL)
        {
            struct local_memory* segment = (struct local_memory*) md->segment;

            _nvm_va_unmap(&md->va_mapping);
            _nvm_io_unmap(&md->io_mapping);
            _nvm_local_memory_put(segment);
            free(segment);
        }

        _nvm_device_put(&md->device);
        free(md);
    }
}



/*
 * Create local segment and map it.
 */
int nvm_dis_dma_create(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, uint32_t adapter, uint32_t id, size_t size)
{
    struct map_descriptor* md;
    *handle = NULL;
    size = NVM_CTRL_ALIGN(ctrl, size);

    // Create mapping descriptor
    int err = create_map_descriptor(&md, ctrl, size);
    if (err != 0)
    {
        return err;
    }

    // Create local segment
    // FIXME: When support is added in SISCI for SCI_FLAG_PRIVATE, remove id argument
    err = create_local_segment((struct local_memory**) &md->segment, id, size, NULL);
    if (err != 0)
    {
        remove_map_descriptor(md);
        return err;
    }

    // Map segment for device
    struct local_memory* sd = (struct local_memory*) md->segment;
    err = _nvm_io_map_local_memory(&md->io_mapping, &md->device, sd, adapter);
    if (err != 0)
    {
        remove_local_segment(md);
        return err;
    }

    // Map segment into virtual memory
    err = _nvm_va_map_local(&md->va_mapping, size, sd->segment);
    if (err != 0)
    {
        remove_local_segment(md);
        return err;
    }
    md->dma_mapping.vaddr = (void*) md->va_mapping.vaddr;

    // Create handle container
    err = _nvm_dma_create(handle, ctrl, (struct dma_map*) md, (dma_map_free_t) remove_local_segment);
    if (err != 0)
    {
        remove_local_segment(md);
        return err;
    }

    // Initialize DMA handle
    _nvm_dma_handle_populate(*handle, ctrl, (uint64_t*) &md->io_mapping.ioaddr);

    return 0;
}



int nvm_dis_dma_map_host(nvm_dma_t** map, const nvm_ctrl_t* ctrl, uint32_t adapter, uint32_t id, void* vaddr, size_t size)
{
    // TODO: add support for _nvm_local_memory_get_registered similar to _nvm_local_memory_get_attached in create_segment()
    dprintf("Function not implemented\n");
    return ENOTSUP;
}



#ifdef _CUDA
/*
 * Map CUDA device memory for device.
 */
int nvm_dis_dma_map_device(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, uint32_t adapter, uint32_t id, void* dptr, size_t size)
{
    struct map_descriptor* md;
    *handle = NULL;
    //size = NVM_CTRL_ALIGN(ctrl, size);
    size = NVM_PAGE_ALIGN(size, (1ULL << 16));

    // Create mapping descriptor
    int err = create_map_descriptor(&md, ctrl, size);
    if (err != 0)
    {
        return err;
    }

    // Create local segment
    err = create_local_segment((struct local_memory**) &md->segment, id, size, dptr);
    if (err != 0)
    {
        remove_map_descriptor(md);
        return err;
    }

    // Map segment for device
    struct local_memory* sd = (struct local_memory*) md->segment;
    err = _nvm_io_map_local_memory(&md->io_mapping, &md->device, sd, adapter);
    if (err != 0)
    {
        remove_local_segment(md);
        return err;
    }

    // Create handle container
    err = _nvm_dma_create(handle, ctrl, (struct dma_map*) md, (dma_map_free_t) remove_local_segment);
    if (err != 0)
    {
        remove_local_segment(md);
        return err;
    }

    // Initialize DMA handle
    _nvm_dma_handle_populate(*handle, ctrl, (uint64_t*) &md->io_mapping.ioaddr);

    return 0;
}
#endif



/*
 * Helper function to unmap and remove local segment.
 */
static void disconnect_remote_segment(struct map_descriptor* md)
{
    if (md != NULL)
    {
        if (md->segment != NULL)
        {
            struct device_memory* segment = (struct device_memory*) md->segment;

            _nvm_va_unmap(&md->va_mapping);
            _nvm_io_unmap(&md->io_mapping);
            _nvm_device_memory_put(segment);
            free(segment);
        }

        _nvm_device_put(&md->device);
        free(md);
    }
}



/*
 * Connect to device memory.
 */
int nvm_dis_dma_connect(nvm_dma_t** handle, const nvm_ctrl_t* ctrl, uint32_t adapter, uint32_t segno, size_t size, bool shared)
{
    struct map_descriptor* md;
    *handle = NULL;
    size = NVM_CTRL_ALIGN(ctrl, size);

    // Create mapping descriptor
    int err = create_map_descriptor(&md, ctrl, size);
    if (err != 0)
    {
        return err;
    }

    const struct device* dev = _nvm_device_from_ctrl(ctrl);

    // Create local segment
    err = connect_remote_segment((struct device_memory**) &md->segment, dev, adapter, segno, shared);
    if (err != 0)
    {
        remove_map_descriptor(md);
        return err;
    }

    // Map segment for device
    struct device_memory* sd = (struct device_memory*) md->segment;
    err = _nvm_io_map_remote(&md->io_mapping, md->device.device, sd->segment);
    if (err != 0)
    {
        disconnect_remote_segment(md);
        return err;
    }

    // Map segment into virtual memory
    err = _nvm_va_map_remote(&md->va_mapping, size, sd->segment, true, true);
    if (err != 0)
    {
        disconnect_remote_segment(md);
        return err;
    }
    md->dma_mapping.vaddr = (void*) md->va_mapping.vaddr;

    // Create handle container
    err = _nvm_dma_create(handle, ctrl, (struct dma_map*) md, (dma_map_free_t) disconnect_remote_segment);
    if (err != 0)
    {
        disconnect_remote_segment(md);
        return err;
    }

    // Initialize DMA handle
    _nvm_dma_handle_populate(*handle, ctrl, (uint64_t*) &md->io_mapping.ioaddr);

    return 0;
}

