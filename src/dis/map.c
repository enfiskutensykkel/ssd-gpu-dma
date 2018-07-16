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
#include "dis/map.h"
#include "dis/local.h"
#include "dis/device.h"
#include "dprintf.h"
#include <sisci_types.h>
#include <sisci_error.h>
#include <sisci_api.h>



int _nvm_va_map_local(struct va_map* m, size_t size, sci_local_segment_t segment)
{
    sci_error_t err = SCI_ERR_OK;

    m->type = _VA_MAP_NOT_MAPPED;

    m->vaddr = SCIMapLocalSegment(segment, &m->md, 0, size, NULL, 0, &err);

    switch (err)
    {
        case SCI_ERR_OK:
            m->type = _VA_MAP_MAPPED_LOCAL;
            return 0;

        case SCI_ERR_OUT_OF_RANGE:
        case SCI_ERR_SIZE_ALIGNMENT:
        case SCI_ERR_OFFSET_ALIGNMENT:
            return EINVAL;

        default:
            dprintf("Mapping segment into virtual address space failed: %s\n", _SCIGetErrorString(err));
            return EIO;
    }
}


int _nvm_va_map_remote(struct va_map* m, size_t size, sci_remote_segment_t segment, bool write, bool wc)
{
    sci_error_t err = SCI_ERR_OK;
    uint32_t flags = 0;

    flags |= !write ? SCI_FLAG_READONLY_MAP : 0;
    flags |= !wc ? SCI_FLAG_IO_MAP_IOSPACE : 0;

    m->type = _VA_MAP_NOT_MAPPED;

    m->vaddr = SCIMapRemoteSegment(segment, &m->md, 0, size, NULL, flags, &err);

    switch (err)
    {
        case SCI_ERR_OK:
            m->type = _VA_MAP_MAPPED_REMOTE;
            return 0;

        case SCI_ERR_OUT_OF_RANGE:
        case SCI_ERR_SIZE_ALIGNMENT:
        case SCI_ERR_OFFSET_ALIGNMENT:
            return EINVAL;

        case SCI_ERR_NOT_CONNECTED:
            return ENOTCONN;

        default:
            dprintf("Mapping segment into virtual address space failed: %s\n", _SCIGetErrorString(err));
            return EIO;
    }
}



void _nvm_va_unmap(struct va_map* m)
{
    sci_error_t err = SCI_ERR_OK;

    switch (m->type)
    {
        case _VA_MAP_MAPPED_LOCAL:
        case _VA_MAP_MAPPED_REMOTE:
            do
            {
                SCIUnmapSegment(m->md, 0, &err);
            }
            while (err == SCI_ERR_BUSY);
            break;

        case _VA_MAP_NOT_MAPPED:
            // Do nothing
            break;

        default:
            dprintf("Unknown mapping type\n");
            break;
    }

    m->type = _VA_MAP_NOT_MAPPED;

#ifndef NDEBUG
    if (err != SCI_ERR_OK)
    {
        dprintf("Unmapping segment from virtual address space failed: %s\n", _SCIGetErrorString(err));
    }
#endif
}



int _nvm_io_map_local(struct io_map* m, sci_device_t device, sci_local_segment_t segment, uint32_t adapter)
{
    sci_error_t err = SCI_ERR_OK;

    m->type = _IO_MAP_NOT_MAPPED;
    m->lsegment = segment;
    m->rsegment = NULL;
    m->device = device;
    m->adapter = adapter;
    m->ioaddr = 0;

    // FIXME: Remove this call when Lars adds support for segments created with SCI_FLAG_PRIVATE
    SCISetSegmentAvailable(m->lsegment, m->adapter, 0, &err);
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

    SCIMapLocalSegmentForDevice(m->lsegment, m->adapter, m->device, &m->ioaddr, 0, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to map segment for device: %s\n", _SCIGetErrorString(err));
        return EIO;
    }

    m->type = _IO_MAP_MAPPED_LOCAL;
    return 0;
}



int _nvm_io_map_remote(struct io_map* m, sci_device_t device, sci_remote_segment_t segment)
{
    sci_error_t err = SCI_ERR_OK;

    m->type = _IO_MAP_NOT_MAPPED;
    m->lsegment = NULL;
    m->rsegment = segment;
    m->device = device;
    m->adapter = 0;
    m->ioaddr = 0;

    SCIMapRemoteSegmentForDevice(m->rsegment, m->device, &m->ioaddr, 0, 0, &err);
    
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to map segment for device: %s\n", _SCIGetErrorString(err));
        return EIO;
    }

    m->type = _IO_MAP_MAPPED_REMOTE;
    return 0;
}



void _nvm_io_unmap(struct io_map* m)
{
    sci_error_t err = SCI_ERR_OK;

    switch (m->type)
    {
        case _IO_MAP_NOT_MAPPED:
            // Do nothing
            break;

        case _IO_MAP_MAPPED_LOCAL:
            do
            {
                SCIUnmapLocalSegmentForDevice(m->lsegment, m->adapter, m->device, 0, &err);
            }
            while (err == SCI_ERR_BUSY);
            break;

        case _IO_MAP_MAPPED_REMOTE:
            do
            {
                SCIUnmapRemoteSegmentForDevice(m->rsegment, m->device, 0, &err);
            }
            while (err == SCI_ERR_BUSY);
            break;

        default:
            dprintf("Unknown mapping type\n");
            break;
    }

#ifndef NDEBUG
    if (err != SCI_ERR_OK)
    {
        dprintf("Unmapping segment for device failed: %s\n", _SCIGetErrorString(err));
    }
#endif

    m->type = _IO_MAP_NOT_MAPPED;
}



int _nvm_io_map_device_memory(struct io_map* m, const struct device_memory* mem)
{
    return _nvm_io_map_remote(m, mem->device.device, mem->segment);
}



int _nvm_io_map_local_memory(struct io_map* m, const struct device* dev, const struct local_memory* mem, uint32_t adapter)
{
    sci_error_t err = SCI_ERR_OK;

    SCIPrepareSegment(mem->segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to prepare local segment: %s\n", _SCIGetErrorString(err));
        return ENOSPC;
    }

    return _nvm_io_map_local(m, dev->device, mem->segment, adapter);
}

