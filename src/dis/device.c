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
#include "mutex.h"
#include "dis/device.h"
#include "dis/map.h"
#include "ctrl.h"
#include "dma.h"
#include "dprintf.h"
#include <sisci_types.h>
#include <sisci_error.h>
#include <sisci_api.h>



static int query_device(const struct device* dev, sci_smartio_device_info_t* info)
{
    sci_error_t err;
    sci_smartio_query_device_t query;
    
    query.fdid = dev->fdid;
    query.subcommand = SCI_Q_DEVICE_INFO;
    query.data = (void*) info;

    SCIQuery(SCI_Q_DEVICE, (void*) &query, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to query device: %s\n", _SCIGetErrorString(err));
        return EIO;
    }

    return 0;
}



/*
 * Helper function to connect to a device memory segment.
 */
int _nvm_device_memory_get(sci_remote_segment_t* segment, const struct device* dev, uint32_t id, unsigned int memtype)
{
    sci_error_t err;
    sci_remote_segment_t seg = NULL;
    
    *segment = NULL;
    
    SCIConnectDeviceSegment(dev->device, &seg, id, memtype, NULL, NULL, 0, &err);
    if (err == SCI_ERR_OK)
    {
        *segment = seg;
        return 0;
    }

    switch (err)
    {
        case SCI_ERR_API_NOSPC:
            return ENOMEM;

        case SCI_ERR_NOSPC:
            return ENOSPC;

        case SCI_ERR_CONNECTION_REFUSED:
            return ECONNREFUSED;

        case SCI_ERR_NO_SUCH_SEGMENT:
        case SCI_ERR_NO_SUCH_NODEID:
        case SCI_ERR_NO_REMOTE_LINK_ACCESS:
        case SCI_ERR_NODE_NOT_RESPONDING:
        case SCI_ERR_SEGMENT_NOT_CONNECTED:
        case SCI_ERR_NO_LINK_ACCESS:
            return EHOSTUNREACH;

        default:
            dprintf("Failed to connect segment%s\n", SCIGetErrorString(err));
            return ENOMEM;
    }
}



void _nvm_device_memory_put(sci_remote_segment_t* segment)
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



int _nvm_local_memory_get(sci_local_segment_t* segment, uint32_t* adapter, const struct device* dev, size_t size, void* ptr, bool gpu)
{
    int status;
    sci_error_t err;
    sci_smartio_device_info_t info;
    uint32_t flags = SCI_FLAG_AUTO_ID;
    sci_local_segment_t seg = NULL;
    uint32_t adapt;

    *segment = NULL;
    *adapter = 0;

    if (gpu && ptr == NULL)
    {
        return EINVAL;
    }

    if (ptr != NULL)
    {
        flags |= SCI_FLAG_EMPTY;
    }

    // Query device to get a possible adapter
    status = query_device(dev, &info);
    if (status != 0)
    {
        return status;
    }
    
    // Query result contains one possible adapter to reach device, we'll use that
    adapt = info.adapter;

    // Create local segment
    SCICreateSegment(dev->sd, &seg, 512, size, NULL, NULL, flags, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to create local segment: %s\n", _SCIGetErrorString(err));
        return ENOSPC;
    }

    if (ptr != NULL && gpu)
    {
#ifdef _CUDA
        // Attach GPU memory
        SCIAttachPhysicalMemory(0, ptr, 0, size, seg, SCI_FLAG_CUDA_BUFFER, &err);
        if (err != SCI_ERR_OK)
        {
            dprintf("Failed to attach GPU memory to local segment: %s\n", _SCIGetErrorString(err));
            SCIRemoveSegment(seg, 0, &err);
            return EIO;
        }
#else
        SCIRemoveSegment(seg, 0, &err);
        return ENOTSUP;
#endif
    }
    else if (ptr != NULL)
    {
        // Register segment memory
        SCIRegisterSegmentMemory(ptr, size, seg, SCI_FLAG_LOCK_USER_MEM, &err);
        if (err != SCI_ERR_OK)
        {
            dprintf("Failed to register local segment memory: %s\n", _SCIGetErrorString(err));
            SCIRemoveSegment(seg, 0, &err);
            return ENOTTY;
        }
    }

    // Export segment on chosen adapter
    SCIPrepareSegment(seg, adapt, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to prepare local segment on adapter %u (ptr=%p, size=%zx): %s\n", 
                adapt, ptr, size, _SCIGetErrorString(err));
        SCIRemoveSegment(seg, 0, &err);
        return EIO;
    }

    *segment = seg;
    *adapter = adapt;
    return 0;
}



void _nvm_local_memory_put(sci_local_segment_t* segment)
{
    if (segment != NULL && *segment != NULL)
    {
        sci_error_t err = SCI_ERR_OK;

        do
        {
            SCIRemoveSegment(*segment, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

#ifndef NDEBUG
        if (err != SCI_ERR_OK)
        {
            dprintf("Failed to remove local segment: %s\n", _SCIGetErrorString(err));
        }
#endif

        *segment = NULL;
    }
}



/*
 * Map local segment for device.
 */
static int io_map_local(const struct device* dev, const struct local_segment* ls, uint64_t* ioaddr)
{
    sci_error_t err;
    sci_ioaddr_t addr;
    size_t size;

    *ioaddr = 0;

    size = SCIGetLocalSegmentSize(ls->segment);
    if (size == 0)
    {
        return EINVAL;
    }

    SCISetSegmentAvailable(ls->segment, ls->adapter, 0, &err);
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

    SCIMapLocalSegmentForDevice(ls->segment, ls->adapter, dev->device, &addr, 0, size, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to map local segment for device: %s\n", _SCIGetErrorString(err));
        return EIO;
    }

    *ioaddr = (uint64_t) addr;
    return 0;
}



/*
 * Map remote segment for device
 */
static int io_map_remote(const struct device* dev, const struct remote_segment* rs, uint64_t* ioaddr)
{
    sci_error_t err;
    sci_ioaddr_t addr;
    size_t size;

    *ioaddr = 0;

    size = SCIGetRemoteSegmentSize(rs->segment);
    if (size == 0)
    {
        return EINVAL;
    }

    SCIMapRemoteSegmentForDevice(rs->segment, dev->device, &addr, 0, size, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to map remote segment for device: %s\n", _SCIGetErrorString(err));
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

    status = _nvm_mutex_init(&dev->lock);
    if (status != 0)
    {
        free(dev);
        return status;
    }

    dev->fdid = fdid;
    dev->sd = NULL;
    dev->counter = 0;
    dev->device = NULL;
    dev->segment = NULL;
    dev->size = 0;
    dev->ptr = NULL;
    dev->md = NULL;

    SCIOpen(&dev->sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        _nvm_mutex_free(&dev->lock);
        free(dev);
        dprintf("Failed to create SISCI virtual device: %s\n", _SCIGetErrorString(err));
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
        _nvm_mutex_free(&dev->lock);
        free(dev);
        return status;
    }

    status = _nvm_device_memory_get(&dev->segment, dev, 0, SCI_MEMTYPE_BAR);
    if (status != 0)
    {
        SCIReturnDevice(dev->device, 0, &err);
        SCIClose(dev->sd, 0, &err);
        _nvm_mutex_free(&dev->lock);
        free(dev);
        return status;
    }

    dev->size = SCIGetRemoteSegmentSize(dev->segment);

    dev->ptr = SCIMapRemoteSegment(dev->segment, &dev->md, 0, dev->size, NULL, SCI_FLAG_IO_MAP_IOSPACE, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to map device memory into local address space: %s\n", _SCIGetErrorString(err));
        _nvm_device_memory_put(&dev->segment);
        SCIReturnDevice(dev->device, 0, &err);
        SCIClose(dev->sd, 0, &err);
        _nvm_mutex_free(&dev->lock);
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

    _nvm_device_memory_put(&dev->segment);

    SCIReturnDevice(dev->device, 0, &err);
    SCIClose(dev->sd, 0, &err);
    _nvm_mutex_free(&dev->lock);
    free(dev);
}



/*
 * Dispatch the proper mapping function based on the type.
 */
static int io_map(const struct device* dev, const struct va_range* va, uint64_t* ioaddr)
{
#ifndef NDEBUG
    // SISCI segments, being contiguous memory, should
    // be considered one large page rather than many pages
    if (va->n_pages != 1)
    {
        dprintf("Virtual address range must be a SISCI segment\n");
        return EINVAL;
    }
#endif

    if (va->remote)
    {
        return io_map_remote(dev, _nvm_container_of(va, struct remote_segment, range), ioaddr);
    }
    else
    {
        return io_map_local(dev, _nvm_container_of(va, struct local_segment, range), ioaddr);
    }
}



static void io_unmap(const struct device* dev, const struct va_range* va)
{
    sci_error_t err = SCI_ERR_OK;
    
    if (va->remote)
    {
        const struct remote_segment* rs;
        rs = _nvm_container_of(va, struct remote_segment, range);

        do
        {
            SCIUnmapRemoteSegmentForDevice(rs->segment, dev->device, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

#ifndef NDEBUG
        if (err != SCI_ERR_OK)
        {
            dprintf("Unmapping remote segment for device failed: %s\n", _SCIGetErrorString(err));
        }
#endif
    }
    else
    {
        const struct local_segment* ls;
        ls = _nvm_container_of(va, struct local_segment, range);

        do
        {
            SCIUnmapLocalSegmentForDevice(ls->segment, ls->adapter, dev->device, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

#ifndef NDEBUG
        if (err != SCI_ERR_OK)
        {
            dprintf("Unmapping local segment for device failed: %s\n", _SCIGetErrorString(err));
        }
#endif

        do
        {
            SCISetSegmentUnavailable(ls->segment, ls->adapter, 0, &err);
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



uint32_t nvm_dis_node_from_ctrl(const nvm_ctrl_t* ctrl)
{
    int err;
    struct controller* c;
    sci_smartio_device_info_t info;

    if (ctrl == NULL)
    {
        return 0;
    }

    if (_nvm_ctrl_type(ctrl) != DEVICE_TYPE_SMARTIO)
    {
        return 0;
    }

    c = _nvm_ctrl_get(ctrl);
    if (c == NULL)
    {
        return 0;
    }

    err = query_device(c->device, &info);
    _nvm_ctrl_put(c);

    if (err == 0)
    {
        return info.nodeid;
    }

    return 0;
}

