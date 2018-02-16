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

using error = std::runtime_error;
using std::string;



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
    return createDeviceDma(ctrl, size);
}
#endif

