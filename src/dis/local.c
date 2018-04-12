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
#include "dis/local.h"
#include "dprintf.h"
#include <sisci_types.h>
#include <sisci_error.h>
#include <sisci_api.h>



/*
 * Helper function to create a local segment.
 */
static int create_segment(struct local_memory* m, uint32_t id, size_t size, uint32_t flags)
{
    sci_error_t err;
    sci_error_t status = SCI_ERR_OK;

    m->id = id;
    m->size = size;
    m->flags = flags;

    SCIOpen(&m->sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to open SISCI descriptor: %s\n", _SCIGetErrorString(err));
        return EIO;
    }

    SCICreateSegment(m->sd, &m->segment, id, size, NULL, NULL, flags, &err);
    if (err != SCI_ERR_OK)
    {
        status = err;
        SCIClose(m->sd, 0, &err);
    }

    switch (status)
    {
        case SCI_ERR_OK:
            return 0;

        case SCI_ERR_SEGMENTID_USED:
            return EEXIST;

        default:
            dprintf("Unknown error while creating local segment: %s\n", _SCIGetErrorString(status));
            return EIO;
    }
}



void _nvm_local_memory_put(struct local_memory* mem)
{
    sci_error_t err = SCI_ERR_OK;

    do
    {
        SCIRemoveSegment(mem->segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

#ifndef NDEBUG
    if (err != SCI_ERR_OK)
    {
        dprintf("Unknown error while removing local segment: %s\n", _SCIGetErrorString(err));
    }
#endif
}



int _nvm_local_memory_get(struct local_memory* mem, uint32_t id, size_t size)
{
    int err;

    // FIXME: When support is added, remove id argument and pass SCI_FLAG_PRIVATE
    err = create_segment(mem, id, size, 0);
    //err = create_segment(mem, id, size, SCI_FLAG_PRIVATE);
    if (err != 0)
    {
        return err;
    }

    return 0;
}



int _nvm_local_memory_get_attached(struct local_memory* mem, uint32_t id, void* ptr, size_t size)
{
    sci_error_t err;
    int status;

    // FIXME: When support is added, use SCI_FLAG_PRIVATE
    //status = create_segment(mem, id, size, SCI_FLAG_EMPTY | SCI_FLAG_PRIVATE);
    status = create_segment(mem, id, size, SCI_FLAG_EMPTY);
    if (status != 0)
    {
        return status;
    }

    SCIAttachPhysicalMemory(0, ptr, 0, size, mem->segment, SCI_FLAG_CUDA_BUFFER, &err);
    if (err != SCI_ERR_OK)
    {
        _nvm_local_memory_put(mem);
        dprintf("Failed to attach memory to local segment: %s\n", _SCIGetErrorString(err));
        return EIO;
    }

    return 0;
}



int _nvm_local_memory_get_registered(struct local_memory* mem, uint32_t id, void* ptr, size_t size)
{
    sci_error_t err;
    int status;

    // FIXME: Add support for SCI_FLAG_PRIVATE
    status = create_segment(mem, id, size, SCI_FLAG_EMPTY /* | SCI_FLAG_PRIVATE */);
    if (status != 0)
    {
        return status;
    }

    SCIRegisterSegmentMemory(ptr, size, mem->segment, SCI_FLAG_LOCK_USER_MEM, &err);
    
    switch (err)
    {
        case SCI_ERR_OK:
            // Do nothing
            break;

        case SCI_ERR_SIZE_ALIGNMENT:
            _nvm_local_memory_put(mem);
            return EINVAL;

        case SCI_ERR_OUT_OF_RANGE:
            _nvm_local_memory_put(mem);
            return ERANGE;

        case SCI_ERR_ILLEGAL_ADDRESS:
            _nvm_local_memory_put(mem);
            return EPERM;

        case SCI_ERR_NOT_SUPPORTED:
            _nvm_local_memory_put(mem);
            dprintf("Function is not supported on platform\n");
            return ENOTSUP;

        default:
            _nvm_local_memory_put(mem);
            dprintf("Unknown error when registering host memory: %s\n", _SCIGetErrorString(err));
            return EIO;
    }

    return 0;
}

