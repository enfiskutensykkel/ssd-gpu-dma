#ifndef _SISCI
#error "Must compile with SISCI support"
#endif

#ifndef __DIS_CLUSTER__
#define __DIS_CLUSTER__
#endif

#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_util.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <errno.h>
#include "dis/device.h"
#include "dis/map.h"
#include "ctrl.h"
#include "dma.h"
#include "dprintf.h"
#include <sisci_types.h>
#include <sisci_error.h>
#include <sisci_api.h>



/*
 * Helper function to connect to a device memory segment.
 */
int _nvm_connect_device_memory(sci_remote_segment_t* segment, const struct device* dev, uint32_t id, unsigned int memtype)
{
    sci_error_t err;
    
    SCIConnectDeviceSegment(dev->device, segment, id, memtype, NULL, NULL, 0, &err);
    if (err == SCI_ERR_OK)
    {
        return 0;
    }

    *segment = NULL;
    switch (err)
    {
        case SCI_ERR_API_NOSPC:
            dprintf("Could not connect to device memory segment, out of memory\n");
            return ENOMEM;

        case SCI_ERR_NOSPC:
            dprintf("Could not connect to device memory segment, out of resources\n");
            return ENOSPC;

        case SCI_ERR_NO_SUCH_SEGMENT:
            dprintf("Failed to connect to device memory segment, segment not found\n");
            return ENOTTY;

        case SCI_ERR_CONNECTION_REFUSED:
            dprintf("Failed to connect to device memory segment, connection was refused\n");
            return EPERM;

        default:
            dprintf("Failed to get device memory reference: %s\n", _SCIGetErrorString(err));
            return ENOMEM;
    }
}



void _nvm_disconnect_device_memory(sci_remote_segment_t* segment)
{
    if (segment != NULL && *segment != NULL)
    {
        sci_error_t err = SCI_ERR_OK;

        do
        {
            SCIDisconnectSegment(*segment, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

#ifndef NDEBUG
        if (err != SCI_ERR_OK)
        {
            dprintf("Failed to disconnect segment: %s\n", _SCIGetErrorString(err));
        }
#endif

        *segment = NULL;
    }
}



/*
 * Map local segment for device.
 */
static int io_map_local(const struct device* dev, const struct map* map, uint64_t* ioaddr)
{
    sci_error_t err;
    size_t size = SCIGetLocalSegmentSize(map->lseg);
    sci_ioaddr_t addr;

    *ioaddr = 0;

    SCISetSegmentAvailable(map->lseg, map->adapter, 0, &err);
    switch (err)
    {
        case SCI_ERR_OK:
            break;

        case SCI_ERR_ILLEGAL_OPERATION:
        case SCI_ERR_SEGMENT_NOT_PREPARED:
            return EINVAL;

        default:
            dprintf("Failed to set segment available: %s\n", _SCIGetErrorString(err));
            return EIO;
    }

    SCIMapLocalSegmentForDevice(map->lseg, map->adapter, dev->device, &addr, 0, size, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to map segment for device: %s\n", _SCIGetErrorString(err));
        return EIO;
    }

    *ioaddr = (uint64_t) addr;
    return 0;
}



/*
 * Map remote segment for device
 */
static int io_map_remote(const struct device* dev, const struct map* map, uint64_t* ioaddr)
{
    sci_error_t err;
    size_t size;
    sci_ioaddr_t addr;

    *ioaddr = 0;

    size = SCIGetRemoteSegmentSize(map->rseg);

    SCIMapRemoteSegmentForDevice(map->rseg, dev->device, &addr, 0, size, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to map segment for device: %s\n", _SCIGetErrorString(err));
        return EIO;
    }

    *ioaddr = (uint64_t) addr;
    return 0;
}



/*
 * Take device reference ("borrow" device) and map MLBAR/BAR0
 */
static int borrow_device(struct device** handle, uint32_t fdid)
{
    int status;
    sci_error_t err;
    struct device* dev;

    *handle = NULL;
    dev = (struct device*) malloc(sizeof(struct device));
    if (dev == NULL)
    {
        dprintf("Failed to allocate device handle: %s\n", strerror(errno));
        return ENOMEM;
    }

    dev->fdid = fdid;
    dev->sd = NULL;
    dev->device = NULL;
    dev->segment = NULL;
    dev->size = 0;
    dev->ptr = NULL;
    dev->md = NULL;

    SCIOpen(&dev->sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to create SISCI virtual device: %s\n", _SCIGetErrorString(err));
        free(dev);
        return ENOSYS;
    }

    SCIBorrowDevice(dev->sd, &dev->device, dev->fdid, 0, &err);
    if (err != SCI_ERR_OK) 
    {
        status = ENODEV;
        dprintf("Failed to increase device reference: %s\n", _SCIGetErrorString(err));
        switch (err)
        {
            case SCI_ERR_NO_SUCH_FDID:
                status = ENODEV;
                break;


            case SCI_ERR_NOT_AVAILABLE:
                status = EBUSY;
                break;

            default:
                status = EIO;
                break;
        }

        SCIClose(dev->sd, 0, &err);
        free(dev);
        return status;
    }

    status = _nvm_connect_device_memory(&dev->segment, dev, 0, SCI_MEMTYPE_BAR);
    if (status != 0)
    {
        SCIReturnDevice(dev->device, 0, &err);
        SCIClose(dev->sd, 0, &err);
        free(dev);
        return status;
    }

    dev->size = SCIGetRemoteSegmentSize(dev->segment);

    dev->ptr = SCIMapRemoteSegment(dev->segment, &dev->md, 0, dev->size, NULL, SCI_FLAG_IO_MAP_IOSPACE, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to map device memory into local address space: %s\n", _SCIGetErrorString(err));
        _nvm_disconnect_device_memory(&dev->segment);
        SCIReturnDevice(dev->device, 0, &err);
        SCIClose(dev->sd, 0, &err);
        free(dev);
        return ENOSPC;
    }
    
    *handle = dev;
    return 0;
}



/*
 * Release device reference ("return" device)
 */
static void return_device(struct device* dev, volatile void* mm_ptr, size_t mm_size)
{
    sci_error_t err;

    // Suppress warnings about unused parameters
    if (mm_ptr != dev->ptr || mm_size != dev->size)
    {
        dprintf("Something is not right\n");
    }

    do
    {
        SCIUnmapSegment(dev->md, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    _nvm_disconnect_device_memory(&dev->segment);

    SCIReturnDevice(dev->device, 0, &err);
    SCIClose(dev->sd, 0, &err);
    free(dev);
}



/*
 * Dispatch the proper mapping function based on the type.
 */
static int io_map(const struct device* dev, const struct va_range* va, uint64_t* ioaddr)
{
    const struct map* map = _nvm_container_of(va, struct map, range);

#ifndef NDEBUG
    // SISCI segments, being contiguous memory, should
    // be considered one large page rather than many pages
    if (va->n_pages != 1)
    {
        dprintf("Virtual address range must be a SISCI segment\n");
        return EINVAL;
    }
#endif

    switch (map->type)
    {
        case SEGMENT_TYPE_LOCAL:
        case SEGMENT_TYPE_PHYSICAL:
            return io_map_local(dev, map, ioaddr);

        case SEGMENT_TYPE_REMOTE:
        case SEGMENT_TYPE_DEVICE:
            return io_map_remote(dev, map, ioaddr);

        default:
            dprintf("Unknown segment type\n");
            return EINVAL;
    }
}



static void io_unmap(const struct device* dev, const struct va_range* va)
{
    sci_error_t err = SCI_ERR_OK;
    const struct map* map = _nvm_container_of(va, struct map, range);

    switch (map->type)
    {
        case SEGMENT_TYPE_LOCAL:
        case SEGMENT_TYPE_PHYSICAL:
            do
            {
                SCIUnmapLocalSegmentForDevice(map->lseg, map->adapter, dev->device, 0, &err);
            }
            while (err == SCI_ERR_BUSY);
            break;

        case SEGMENT_TYPE_REMOTE:
        case SEGMENT_TYPE_DEVICE:
            do
            {
                SCIUnmapRemoteSegmentForDevice(map->rseg, dev->device, 0, &err);
            }
            while (err == SCI_ERR_BUSY);
            break;

        default:
            dprintf("Unknown segment type\n");
            return;
    }

#ifndef NDEBUG
    if (err != SCI_ERR_OK)
    {
        dprintf("Unmapping segment for device failed: %s\n", _SCIGetErrorString(err));
    }
#endif

    if (map->type == SEGMENT_TYPE_PHYSICAL)
    {
        do
        {
            SCISetSegmentUnavailable(map->lseg, map->adapter, 0, &err);
        }
        while (err == SCI_ERR_BUSY);
    }
}



/*
 * Device operations
 */
static const struct device_ops smartio_device_ops = 
{
    .release_device = &return_device,
    .map_range = &io_map,
    .unmap_range = &io_unmap,
};



int nvm_dis_ctrl_init(nvm_ctrl_t** ctrl, uint32_t fdid)
{
    int err;
    struct device* dev;

    *ctrl = NULL;

    err = borrow_device(&dev, fdid);
    if (err != 0)
    {
        return err;
    }

    err = _nvm_ctrl_init(ctrl, dev, &smartio_device_ops, DEVICE_TYPE_SMARTIO, dev->ptr, dev->size);
    if (err != 0)
    {
        return_device(dev, dev->ptr, dev->size);
        return err;
    }

    return 0;
}



void nvm_dis_ctrl_unmap_p2p_device(const nvm_ctrl_t* ctrl, sci_smartio_device_t dev)
{
    if (ctrl != NULL)
    {
        const struct controller* container = _nvm_container_of(ctrl, struct controller, handle);
            
        if (container->device != NULL && container->type == DEVICE_TYPE_SMARTIO)
        {
            sci_error_t err;

            do
            {
                SCIUnmapRemoteSegmentForDevice(container->device->segment, dev, 0, &err);
            }
            while (err == SCI_ERR_BUSY);
        }
    }
}



int nvm_dis_ctrl_map_p2p_device(const nvm_ctrl_t* ctrl, sci_smartio_device_t dev, uint64_t* ioaddr)
{
    sci_error_t err;
    sci_ioaddr_t addr;
    const struct controller* container;

    if (ctrl == NULL)
    {
        return EINVAL;
    }

    container = _nvm_container_of(ctrl, struct controller, handle);
    if (container->device == NULL || container->type != DEVICE_TYPE_SMARTIO)
    {
        return EINVAL;
    }

    SCIMapRemoteSegmentForDevice(container->device->segment, dev, &addr, 0, container->device->size, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to map controller BAR for device: %s\n", _SCIGetErrorString(err));
        return EIO;
    }

    if (ioaddr != NULL)
    {
        *ioaddr = (uint64_t) addr;
    }

    return 0;
}

