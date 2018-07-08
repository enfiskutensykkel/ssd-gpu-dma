#include <cuda.h>
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
#include <cstdlib>
#include <cstring>
#include "buffer.h"

#ifdef __DIS_CLUSTER__
#include <sisci_api.h>
#endif

using error = std::runtime_error;
using std::string;


#ifdef __DIS_CLUSTER__
struct SegmentHolder
{
    void*               handle;
    sci_desc_t          descriptor;
    sci_local_segment_t segment;
};


typedef std::shared_ptr<SegmentHolder> SegmentPtr;
#endif



static void* allocateDeviceMemory(size_t size, int device)
{
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to set CUDA device: ") + cudaGetErrorString(err));
    }

    void* ptr = nullptr;
    err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to allocate device memory: ") + cudaGetErrorString(err));
    }

    err = cudaMemset(ptr, 0, size);
    if (err != cudaSuccess)
    {
        free(ptr);
        throw error(string("Failed to clear device memory: ") + cudaGetErrorString(err));
    }

    return ptr;
}



static void* allocateHostMemory(size_t size, uint32_t flags)
{
    void* ptr = nullptr;

    cudaError_t err = cudaHostAlloc(&ptr, size, flags);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to allocate host memory: ") + cudaGetErrorString(err));
    }

    memset(ptr, 0, size);

    return ptr;
}



static void* lookupDevicePointer(void* pointer)
{
    cudaPointerAttributes attrs;

    auto err = cudaPointerGetAttributes(&attrs, pointer);
    if (err != cudaSuccess)
    {
        return nullptr;
    }

    return attrs.devicePointer;
}



#ifdef __DIS_CLUSTER__
static void unmapAndRemoveDeviceMemory(SegmentHolder* s)
{
    sci_error_t err;

    do
    {
        SCIRemoveSegment(s->segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    SCIClose(s->descriptor, 0, &err);
    cudaFree(s->handle);
    delete s;
}
#endif



#ifdef __DIS_CLUSTER__
static SegmentPtr allocateDeviceMemoryAndMap(size_t size, int device, uint32_t id)
{
    void* buffer = allocateDeviceMemory(size, device);
    
    void* devicePointer = lookupDevicePointer(buffer);
    if (devicePointer == nullptr)
    {
        cudaFree(buffer);
        throw error("Failed to look up device pointer");
    }

    SegmentHolder* s = new (std::nothrow) SegmentHolder;
    if (s == nullptr)
    {
        cudaFree(buffer);
        throw error("Failed to allocate segment holder");
    }

    s->handle = buffer;

    sci_error_t err;
    SCIOpen(&s->descriptor, 0, &err);
    if (err != SCI_ERR_OK)
    {
        cudaFree(buffer);
        delete s;
        throw error("Failed to open SISCI descriptor: " + string(SCIGetErrorString(err)));
    }
    
    SCICreateSegment(s->descriptor, &s->segment, id, size, nullptr, nullptr, SCI_FLAG_EMPTY, &err);
    if (err != SCI_ERR_OK)
    {
        auto fail = err;
        SCIClose(s->descriptor, 0, &err);
        cudaFree(buffer);
        delete s;
        throw error("Failed to create segment: " + string(SCIGetErrorString(fail)));
    }

    SCIAttachPhysicalMemory(0, devicePointer, 0, size, s->segment, SCI_FLAG_CUDA_BUFFER, &err);
    if (err != SCI_ERR_OK)
    {
        auto fail = err;
        SCIRemoveSegment(s->segment, 0, &err);
        SCIClose(s->descriptor, 0, &err);
        cudaFree(buffer);
        delete s;
        throw error("Failed to create segment: " + string(SCIGetErrorString(fail)));
    }

    return SegmentPtr(s, unmapAndRemoveDeviceMemory);
}
#endif



MemPtr createDeviceMemory(size_t size, int cudaDevice)
{
    void* pointer = allocateDeviceMemory(size, cudaDevice);
    return MemPtr(pointer, cudaFree);
}



MemPtr createHostMemory(size_t size, uint32_t flags)
{
    void* pointer = allocateHostMemory(size, flags);
    return MemPtr(pointer, cudaFreeHost);
}



MemPtr createHostMemory(size_t size)
{
    return createHostMemory(size, cudaHostAllocDefault);
}



DmaPtr createHostDma(const nvm_ctrl_t* ctrl, size_t size)
{
    void* pointer = allocateHostMemory(size, cudaHostAllocDefault);

    nvm_dma_t* dma = nullptr;
    int err = nvm_dma_map_host(&dma, ctrl, pointer, size);
    if (!nvm_ok(err))
    {
        cudaFreeHost(pointer);
        throw error(string("Failed to map host memory: ") + nvm_strerror(err));
    }

    return DmaPtr(dma, [pointer](nvm_dma_t* dma) {
        nvm_dma_unmap(dma);
        cudaFreeHost(pointer);
    });
}



DmaPtr createDeviceDma(const nvm_ctrl_t* ctrl, size_t size, int device)
{
    void* pointer = allocateDeviceMemory(size, device);

    void* devicePointer = lookupDevicePointer(pointer);
    if (devicePointer == nullptr)
    {
        cudaFree(pointer);
        throw error("Failed to look up device pointer");
    }

    nvm_dma_t* dma = nullptr;
    int err = nvm_dma_map_device(&dma, ctrl, devicePointer, size);
    if (!nvm_ok(err))
    {
        cudaFree(pointer);
        throw error(string("Failed to map device memory: ") + nvm_strerror(err));
    }

    dma->vaddr = pointer;

    return DmaPtr(dma, [pointer](nvm_dma_t* dma) {
        nvm_dma_unmap(dma);
        cudaFree(pointer);
    });
}



#ifdef __DIS_CLUSTER__
DmaPtr createRemoteDma(const nvm_ctrl_t* ctrl, size_t size, uint32_t adapter, uint32_t number)
{
    nvm_dma_t* dma = nullptr;

    int err = nvm_dis_dma_connect(&dma, ctrl, adapter, number, size, true); // FIXME: Should be private

    if (!nvm_ok(err))
    {
        throw error(string("Failed to connect to remote segment: ") + nvm_strerror(err));
    }

    return DmaPtr(dma, nvm_dma_unmap);
}
#else
DmaPtr createRemoteDma(const nvm_ctrl_t* ctrl, size_t size, uint32_t, uint32_t)
{
    return createHostDma(ctrl, size);
}
#endif



#ifdef __DIS_CLUSTER__
DmaPtr createHostDma(const nvm_ctrl_t* ctrl, size_t size, uint32_t adapter, uint32_t id)
{
    nvm_dma_t* dma = nullptr;

    int err = nvm_dis_dma_create(&dma, ctrl, adapter, id, size);
    if (!nvm_ok(err))
    {
        throw error(string("Failed to create local segment: ") + nvm_strerror(err));
    }

    return DmaPtr(dma, nvm_dma_unmap);
}
#else
DmaPtr createHostDma(const nvm_ctrl_t* ctrl, size_t size, uint32_t, uint32_t)
{
    return createHostDma(ctrl, size);
}
#endif



#ifdef __DIS_CLUSTER__
DmaPtr createDeviceDma(const nvm_ctrl_t* ctrl, size_t size, int device, uint32_t adapter, uint32_t id)
{
    void* pointer = allocateDeviceMemory(size, device);

    void* devicePointer = lookupDevicePointer(pointer);
    if (devicePointer == nullptr)
    {
        cudaFree(pointer);
        throw error("Failed to look up device pointer");
    }

    nvm_dma_t* dma = nullptr;
    int err = nvm_dis_dma_map_device(&dma, ctrl, adapter, id, devicePointer, size);
    if (!nvm_ok(err))
    {
        cudaFree(pointer);
        throw error(string("Failed to map device memory: ") + nvm_strerror(err));
    }

    dma->vaddr = pointer;

    return DmaPtr(dma, [pointer](nvm_dma_t* dma) {
        nvm_dma_unmap(dma);
        cudaFree(pointer);
    });
}
#else
DmaPtr createDeviceDma(const nvm_ctrl_t* ctrl, size_t size, int device, uint32_t, uint32_t)
{
    return createDeviceDma(ctrl, size, device);
}
#endif



#ifdef __DIS_CLUSTER__
DmaPtr createDeviceDmaMapped(const nvm_ctrl_t* ctrl, size_t size, int device, uint32_t adapter, uint32_t id)
{
    SegmentPtr segment = allocateDeviceMemoryAndMap(size, device, id);

    sci_error_t err;
    SCIPrepareSegment(segment->segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        throw error("Failed to prepare segment on adapter: " + string(SCIGetErrorString(err)));
    }

    SCISetSegmentAvailable(segment->segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        throw error("Failed to set segment avaialable: " + string(SCIGetErrorString(err)));
    }

    nvm_dma_t* dma = nullptr;
    int status = nvm_dis_dma_map_local(&dma, ctrl, adapter, segment->segment, true);
    if (!nvm_ok(status))
    {
        throw error("Failed to map device memory segment for device: " + string(nvm_strerror(status)));
    }

    return DmaPtr(dma, [segment, adapter](nvm_dma_t* dma) mutable {
        nvm_dma_unmap(dma);

        sci_error_t err;
        do
        {
            SCISetSegmentUnavailable(segment->segment, adapter, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

        segment.reset();
    });
}
#endif
