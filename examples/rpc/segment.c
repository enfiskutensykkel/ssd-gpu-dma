#include <stddef.h>
#include <stdint.h>
#include <errno.h>
#include <nvm_types.h>
#include <nvm_dma.h>
#include <nvm_util.h>
#include <sisci_types.h>
#include <sisci_api.h>
#include <sisci_error.h>
#include "segment.h"


int segment_create(struct segment* segment, uint32_t segment_id, size_t size)
{
    sci_error_t err;
    sci_error_t status;

    SCIOpen(&segment->sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        return EIO;
    }

    SCICreateSegment(segment->sd, &segment->segment, segment_id, size, NULL, NULL, 0, &err);
    if (err != SCI_ERR_OK)
    {
        SCIClose(segment->sd, 0, &status);

        if (err == SCI_ERR_SEGMENTID_USED)
        {
            return EEXIST;
        }

        return ENOSPC;
    }

    segment->id = segment_id;
    segment->size = size;
    return 0;
}


void segment_remove(struct segment* segment)
{
    sci_error_t err;

    do
    {
        SCIRemoveSegment(segment->segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    SCIClose(segment->sd, 0, &err);
}


int dma_create(nvm_dma_t** window, const nvm_ctrl_t* ctrl, struct segment* segment, uint32_t adapter)
{
    sci_error_t err;

    SCIPrepareSegment(segment->segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        return ENOSPC;
    }

    SCISetSegmentAvailable(segment->segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        return EIO;
    }

    int status = nvm_dis_dma_map_local(window, ctrl, adapter, segment->segment, true);
    if (status != 0)
    {
        do
        {
            SCISetSegmentUnavailable(segment->segment, adapter, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

        return status;
    }

    return 0;
}


void dma_remove(nvm_dma_t* window, struct segment* segment, uint32_t adapter)
{
    sci_error_t err;

    nvm_dma_unmap(window);

    do
    {
        SCISetSegmentUnavailable(segment->segment, adapter, 0, &err);
    }
    while (err == SCI_ERR_BUSY);
}

