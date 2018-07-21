#include <nvm_dma.h>
#include <nvm_error.h>
#include <sisci_types.h>
#include <sisci_api.h>
#include "buffer.h"
#include "map.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <cstdio>

#define CUDA_DEVICE_BAR_NO  1

using error = std::runtime_error;
using std::string;


struct Segment
{
    MemPtr                  buffer;
    void*                   devicePointer;
    size_t                  size;
    size_t                  offset;
    uint64_t                localIoAddr;
    uint64_t                remoteIoAddr;
    sci_desc_t              sd;
    sci_device_t            deviceHandle;
    uint32_t                deviceAdapter;
    sci_local_segment_t     localHandle;
    sci_remote_segment_t    connectedBar;
    sci_map_t               mapping;
    volatile void*          addr;
};



static sci_error_t calculateOffset(Segment* segment)
{
    sci_error_t err;

    sci_query_local_segment_t localQuery;
    localQuery.subcommand = SCI_Q_LOCAL_SEGMENT_PHYS_ADDR;
    localQuery.segment = segment->localHandle;

    SCIQuery(SCI_Q_LOCAL_SEGMENT, &localQuery, 0, &err);
    if (err != SCI_ERR_OK)
    {
        return err;
    }

    sci_query_remote_segment_t remoteQuery;
    remoteQuery.subcommand = SCI_Q_REMOTE_SEGMENT_IOADDR;
    remoteQuery.segment = segment->connectedBar;

    SCIQuery(SCI_Q_REMOTE_SEGMENT, &remoteQuery, 0, &err);
    if (err != SCI_ERR_OK)
    {
        return err;
    }

    uint64_t localAddr = localQuery.data.ioaddr;
    uint64_t remoteAddr = remoteQuery.data.ioaddr;

    segment->offset = remoteAddr - localAddr;
    
    return SCI_ERR_OK;
}



static void releaseSegment(Segment* segment)
{
    sci_error_t err;

    do
    {
        SCIUnmapSegment(segment->mapping, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    do
    {
        SCIDisconnectSegment(segment->connectedBar, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    do
    {
        SCIRemoveSegment(segment->localHandle, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    SCIReturnDevice(segment->deviceHandle, 0, &err);
    SCIClose(segment->sd, 0, &err);
    delete segment;
}



SegPtr createGpuMapping(MemPtr gpuBuffer, size_t size, uint64_t gpuDeviceId, uint32_t adapter, uint32_t segmentId)
{
    size = NVM_PAGE_ALIGN(size, 0x10000);
    sci_error_t err;
    string errStr;
    Segment* segment = new Segment;
    segment->buffer = gpuBuffer;
    segment->size = size;
    segment->deviceAdapter = adapter;
    segment->offset = 0;
    segment->localIoAddr = 0;
    segment->remoteIoAddr = 0;
    segment->devicePointer = lookupDevicePointer(gpuBuffer);

    if (segment->devicePointer == nullptr)
    {
        throw error("Failed to look up device pointer");
    }

    SCIOpen(&segment->sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        errStr = "Failed to open SISCI descriptor: " + string(SCIGetErrorString(err));
        goto freeSegment;
    }

    SCIBorrowDevice(segment->sd, &segment->deviceHandle, gpuDeviceId, 0, &err);
    if (err != SCI_ERR_OK)
    {
        errStr = "Failed to borrow device: " + string(SCIGetErrorString(err));
        goto closeDesc;
    }

    SCICreateSegment(segment->sd, &segment->localHandle, segmentId, size, nullptr, nullptr, SCI_FLAG_EMPTY, &err);
    if (err != SCI_ERR_OK)
    {
        errStr = "Failed to create empty local segment: " + string(SCIGetErrorString(err));
        goto returnDevice;
    }

    SCIAttachPhysicalMemory(0, segment->devicePointer, 0, size, segment->localHandle, SCI_FLAG_CUDA_BUFFER, &err);
    if (err != SCI_ERR_OK)
    {
        errStr = "Failed to attach memory to local segment: " + string(SCIGetErrorString(err));
        goto removeSegment;
    }

    SCIPrepareSegment(segment->localHandle, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        errStr = "Failed to prepare local segment on adapter: " + string(SCIGetErrorString(err));
    }

    SCIConnectDeviceMemory(segment->sd, &segment->connectedBar, adapter, segment->deviceHandle, CUDA_DEVICE_BAR_NO, 0, SCI_FLAG_BAR, &err);
    if (err != SCI_ERR_OK)
    {
        errStr = "Failed to connect to BAR memory: " + string(SCIGetErrorString(err));
        goto removeSegment;
    }

    segment->addr = SCIMapRemoteSegment(segment->connectedBar, &segment->mapping, 0, segment->size, nullptr, SCI_FLAG_IO_MAP_IOSPACE, &err);
    if (err != SCI_ERR_OK)
    {
        errStr = "Failed to map BAR memory: " + string(SCIGetErrorString(err));
        goto disconnect;
    }

    err = calculateOffset(segment);
    if (err != SCI_ERR_OK)
    {
        errStr = "Could not query segments: " + string(SCIGetErrorString(err));
        goto unmap;
    }

    return SegPtr(segment, [gpuBuffer](Segment* segment) mutable {
        releaseSegment(segment);
        gpuBuffer.reset();
    });

unmap:
    SCIUnmapSegment(segment->mapping, 0, &err);

disconnect:
    SCIDisconnectSegment(segment->connectedBar, 0, &err);

removeSegment:
    SCIRemoveSegment(segment->localHandle, 0, &err);

returnDevice:
    SCIReturnDevice(segment->deviceHandle, 0, &err);

closeDesc:
    SCIClose(segment->sd, 0, &err);

freeSegment:
    delete segment;
    throw error(errStr); 
}



DmaPtr createDeviceDma(const nvm_ctrl_t* ctrl, uint32_t adapter, SegPtr segment)
{
    sci_error_t err;
    sci_error_t err2;

    SCIPrepareSegment(segment->localHandle, adapter, 0, &err2);
    /* ignore error */

    SCISetSegmentAvailable(segment->localHandle, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        throw error("Failed to set segment available on adapter: " + string(SCIGetErrorString(err)));
    }

    nvm_dma_t* dma = nullptr;
    auto status = nvm_dis_dma_map_local(&dma, ctrl, adapter, segment->localHandle, false);
    if (!nvm_ok(status)) 
    {
        SCISetSegmentUnavailable(segment->localHandle, adapter, 0, &err);
        throw error("Failed to map segment for controller: " + string(nvm_strerror(status)));
    }

    dma->vaddr = (void*) (((unsigned char*) segment->addr) + segment->offset);

    return DmaPtr(dma, [segment, adapter](nvm_dma_t* dma) mutable {
        sci_error_t err;
        nvm_dma_unmap(dma);
        SCISetSegmentUnavailable(segment->localHandle, adapter, 0, &err);
        segment.reset();
    });
}

