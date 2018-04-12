#ifndef _SISCI
#error "Must compile with SISCI support"
#endif

#ifndef __DIS_CLUSTER__
#define __DIS_CLUSTER__
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include "dis/device.h"
#include "dprintf.h"
#include <sisci_types.h>
#include <sisci_error.h>
#include <sisci_api.h>



/*
 * Increase device reference.
 */
int _nvm_device_get(struct device* dev, uint64_t dev_id)
{
    sci_error_t err;

    if (dev == NULL)
    {
        return EINVAL;
    }

    dev->device_id = dev_id;

    SCIOpen(&dev->sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to create virtual device: %s\n", _SCIGetErrorString(err));
        return EIO;
    }

    SCIBorrowDevice(dev->sd, &dev->device, dev_id, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to increase device reference: %s\n", _SCIGetErrorString(err));
        SCIClose(dev->sd, 0, &err);
        return ENODEV;
    }

    return 0;
}



/*
 * Decrease device reference.
 */
void _nvm_device_put(struct device* dev)
{
    sci_error_t err;

    if (dev != NULL)
    {
        SCIReturnDevice(dev->device, 0, &err);
        SCIClose(dev->sd, 0, &err);
    }
}



/*
 * Helper function to connect to a segment.
 */
int _nvm_device_memory_get(struct device_memory* mem,
                           const struct device* dev, 
                           uint32_t adapter,
                           uint32_t segment_no, 
                           uint32_t flags)
{
    sci_error_t err;
    int status = 0;

    status = _nvm_device_get(&mem->device, dev->device_id);
    if (status != 0)
    {
        goto leave;
    }

    const struct device* d = &mem->device;
    mem->adapter = adapter;
    mem->segment_no = segment_no;
    mem->flags = flags;

    SCIConnectDeviceMemory(d->sd, &mem->segment, mem->adapter, d->device, segment_no, 0, flags, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to connect to device memory: %s\n", _SCIGetErrorString(err));
        status = ENODEV;
        goto release;
    }

    return 0;

release:
    _nvm_device_put(&mem->device);

leave:
    return status;
}



/*
 * Disconnect from device memory.
 */
void _nvm_device_memory_put(struct device_memory* mem)
{
    sci_error_t err = SCI_ERR_OK;

    do
    {
        SCIDisconnectSegment(mem->segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

#ifndef NDEBUG
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to disconnect from device memory: %s\n", _SCIGetErrorString(err));
    }
#endif

    _nvm_device_put(&mem->device);
}

