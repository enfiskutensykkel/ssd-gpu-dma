#include "ctrl.h"
#include "gpu.h"
#include "buffer.h"
#include <nvm_types.h>
#include <nvm_dma.h>
#include <nvm_util.h>
#include <nvm_error.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <new>
#include <cstddef>
#include <cstdint>
#include <sisci_types.h>
#include <sisci_error.h>
#include <sisci_api.h>

using error = std::runtime_error;
using DevPtr = std::shared_ptr<Device>;
using std::string;
using std::logic_error;



/*
 * SmartIO device descriptor wrapper.
 */
struct Device
{
    int             cudaDevice;
    uint64_t        fdid;
    uint32_t        adapter;
    sci_desc_t      descriptor;
    sci_device_t    device;
    sci_dev_info_t  info;
};



/*
 * Remote GPU memory reference wrapper.
 */
struct RemoteMemory
{
    Device                  device;
    MemoryPtr               buffer;
    sci_local_segment_t     local;
    sci_remote_segment_t    bar;
    size_t                  size;
    sci_map_t               mapping;
};



static void returnDevice(Device* device)
{
    sci_error_t err;
    SCIReturnDevice(device->device, 0, &err);
    SCIClose(device->descriptor, 0, &err);
}



static void borrowDevice(Device* device, uint64_t fdid, uint32_t adapter, int cudaDevice)
{
    sci_error_t err;

    device->cudaDevice = cudaDevice;
    device->fdid = fdid;
    device->adapter = adapter;

    SCIOpen(&device->descriptor, 0, &err);
    if (err != SCI_ERR_OK)
    {
        throw error("Failed to open descriptor: " + string(SCIGetErrorString(err)));
    }

    SCIBorrowDevice(device->descriptor, &device->device, fdid, 0, &err);
    if (err != SCI_ERR_OK)
    {
        const auto* errstr = SCIGetErrorString(err);
        SCIClose(device->descriptor, 0, &err);
        throw error("Failed to borrow device: " + string(errstr));
    }

    sci_query_dev_t query;
    query.global_devid = fdid;
    query.subcommand = SCI_Q_DEVICE_INFO;
    query.data = (void*) &device->info;

    SCIQuery(SCI_Q_DEVICE, (void*) &query, 0, &err);
    if (err != SCI_ERR_OK)
    {
        const auto* errstr = SCIGetErrorString(err);
        SCIReturnDevice(device->device, 0, &err);
        SCIClose(device->descriptor, 0, &err);
        throw error("Failed to query borrowed device: " + string(errstr));
    }
}



static DevPtr lookupDevice(uint64_t fdid, uint32_t adapter)
{
    Device* device = new (std::nothrow) Device;
    if (device == nullptr)
    {
        throw error("Failed to allocate device descriptor");
    }

    try
    {
        borrowDevice(device, fdid, adapter, -1);
    }
    catch (const error& e)
    {
        delete device;
        throw e;
    }

    // Try to look up device from BDF
    try
    {
        int domain = 0; // FIXME: Implement this
        int bus = (device->info.bdf >> 8) & 0xff;
        int devfn = (device->info.bdf >> 3) & 0x1f;

        device->cudaDevice = Gpu::findDevice(domain, bus, devfn);
    }
    catch (const error& e)
    {
        returnDevice(device);
        delete device;
        throw e;
    }

    return DevPtr(device, [](Device* device) {
        returnDevice(device);
        delete device;
    });
}



Gpu::Gpu(uint64_t fdid, uint32_t adapter)
    : device(lookupDevice(fdid, adapter))
    , cudaDevice(device->cudaDevice)
    , fdid(fdid)
    , adapter(adapter)
{
    // Some sanity checking
    setDevice();
}



DmaPtr Gpu::allocateBuffer(const Ctrl& ctrl, size_t size, uint32_t segmentId) const
{
    if (ctrl.fdid == 0)
    {
        return this->allocateBuffer(ctrl, size);
    }

    auto memory = allocateMemory(size);
    void* pointer = getDevicePointer(memory);

    nvm_dma_t* dma = nullptr;
    int status = nvm_dis_dma_map_device(&dma, ctrl.handle, adapter, segmentId, pointer, size);
    if (!nvm_ok(status))
    {
        throw error("Failed to map device memory for controller: " + string(nvm_strerror(status)));
    }

    dma->vaddr = memory.get();

    return DmaPtr(dma, [memory](nvm_dma_t* dma) mutable {
        nvm_dma_unmap(dma);
        memory.reset();
    });
}



static void releaseReference(RemoteMemory* segment)
{
    sci_error_t err;

    do
    {
        SCIUnmapSegment(segment->mapping, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    do
    {
        SCIDisconnectSegment(segment->bar, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    do
    {
        SCIRemoveSegment(segment->local, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    returnDevice(&segment->device);
    delete segment;
}



static size_t calculateOffset(const RemoteMemory* segment)
{
    sci_error_t err;

    sci_query_local_segment_t localQuery;
    localQuery.subcommand = SCI_Q_LOCAL_SEGMENT_PHYS_ADDR;
    localQuery.segment = segment->local;

    SCIQuery(SCI_Q_LOCAL_SEGMENT, &localQuery, 0, &err);
    if (err != SCI_ERR_OK)
    {
        throw error("Failed to query local segment: " + string(SCIGetErrorString(err)));
    }

    sci_query_remote_segment_t remoteQuery;
    remoteQuery.subcommand = SCI_Q_REMOTE_SEGMENT_IOADDR;
    remoteQuery.segment = segment->bar;

    SCIQuery(SCI_Q_REMOTE_SEGMENT, &remoteQuery, 0, &err);
    if (err != SCI_ERR_OK)
    {
        throw error("Failed to query remote segment: " + string(SCIGetErrorString(err)));
    }

    uint64_t localAddr = localQuery.data.ioaddr;
    uint64_t remoteAddr = remoteQuery.data.ioaddr;

    return localAddr - remoteAddr;
}



DmaPtr Gpu::allocateBufferAndMap(const Ctrl& ctrl, size_t size, uint32_t segmentId) const
{
    if (ctrl.fdid == 0)
    {
        throw logic_error("Controller is not a SmartIO device");
    }

    auto buffer = allocateMemory(size);
    void* devicePointer = getDevicePointer(buffer);
    volatile void* mappedPointer = nullptr;
    size_t offset = 0;
    int adapter = this->adapter; // Capture adapter for later

    RemoteMemory* segment = new (std::nothrow) RemoteMemory;;
    if (segment == nullptr)
    {
        throw error("Failed to allocate remote GPU memory descriptor");
    }

    try
    {
        borrowDevice(&segment->device, fdid, adapter, cudaDevice);
    }
    catch (const error& e)
    {
        delete segment;
        throw e;
    }

    string errStr;
    sci_error_t err;
    int status;
    nvm_dma_t* dma = nullptr;

    SCICreateSegment(segment->device.descriptor, &segment->local, segmentId, size, nullptr, nullptr, SCI_FLAG_EMPTY, &err);
    if (err != SCI_ERR_OK)
    {
        errStr = "Failed to create local segment: " + string(SCIGetErrorString(err));
        goto returnDeviceHandle;
    }

    SCIAttachPhysicalMemory(0, devicePointer, 0, size, segment->local, SCI_FLAG_CUDA_BUFFER, &err);
    if (err != SCI_ERR_OK)
    {
        errStr = "Failed to attach device memory to segment: " + string(SCIGetErrorString(err));
        goto removeSegment;
    }

    SCIPrepareSegment(segment->local, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        errStr = "Failed to prepare segment on adapter: " + string(SCIGetErrorString(err));
        goto removeSegment;
    }

    SCIConnectDeviceMemory(segment->device.descriptor, &segment->bar, adapter, segment->device.device, 1, 0, SCI_FLAG_BAR, &err);
    if (err != SCI_ERR_OK)
    {
        errStr = "Failed to connect to remote device memory: " + string(SCIGetErrorString(err));
        goto removeSegment;
    }

    segment->size = SCIGetRemoteSegmentSize(segment->bar);
    mappedPointer = SCIMapRemoteSegment(segment->bar, &segment->mapping, 0, segment->size, nullptr, /*SCI_FLAG_IO_MAP_IOSPACE*/ 0, &err);
    if (err != SCI_ERR_OK)
    {
        errStr = "Failed to map remote device memory: " + string(SCIGetErrorString(err));
        goto disconnectSegment;
    }

    try
    {
        offset = calculateOffset(segment);
    }
    catch (const error& e)
    {
        errStr = string(e.what());
        goto unmapSegment;
    }

    SCISetSegmentAvailable(segment->local, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        errStr = "Failed to set segment available on adapter: " + string(SCIGetErrorString(err));
        goto unmapSegment;
    }

    status = nvm_dis_dma_map_local(&dma, ctrl.handle, adapter, segment->local, false);
    if (!nvm_ok(status))
    {
        errStr = "Failed to map device memory segment for controller: " + string(nvm_strerror(status));
        goto unavailable;
    }

    dma->vaddr = (void*) (((volatile unsigned char*) mappedPointer) + offset);

    return DmaPtr(dma, [segment, adapter, buffer](nvm_dma_t* dma) mutable {
        sci_error_t err;

        nvm_dma_unmap(dma);

        do
        {
            SCISetSegmentUnavailable(segment->local, adapter, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

        releaseReference(segment);

        buffer.reset();
    });

unavailable:
    do
    {
        SCISetSegmentUnavailable(segment->local, adapter, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

unmapSegment:
    SCIUnmapSegment(segment->mapping, 0, &err);

disconnectSegment:
    SCIDisconnectSegment(segment->bar, 0, &err);

removeSegment:
    SCIRemoveSegment(segment->local, 0, &err);

returnDeviceHandle:
    returnDevice(&segment->device);
    
    delete segment;
    throw error(errStr);
}

