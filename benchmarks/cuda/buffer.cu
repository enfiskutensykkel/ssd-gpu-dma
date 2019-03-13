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
#include "buffer.h"

using error = std::runtime_error;
using std::string;



static void getDeviceMemory(int device, void*& bufferPtr, void*& devicePtr, size_t size)
{
    bufferPtr = nullptr;
    devicePtr = nullptr;

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to set CUDA device: ") + cudaGetErrorString(err));
    }

    err = cudaMalloc(&bufferPtr, size);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to allocate device memory: ") + cudaGetErrorString(err));
    }

    err = cudaMemset(bufferPtr, 0, size);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to clear device memory: ") + cudaGetErrorString(err));
    }

    cudaPointerAttributes attrs;
    err = cudaPointerGetAttributes(&attrs, bufferPtr);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to get pointer attributes: ") + cudaGetErrorString(err));
    }

    devicePtr = attrs.devicePointer;
}



static void getDeviceMemory(int device, void*& bufferPtr, size_t size)
{
    void* notUsed = nullptr;
    getDeviceMemory(device, bufferPtr, notUsed, size);
}



DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size)
{
    nvm_dma_t* dma = nullptr;
    void* buffer = nullptr;

    cudaError_t err = cudaHostAlloc(&buffer, size, cudaHostAllocDefault);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to allocate host memory: ") + cudaGetErrorString(err));
    }

    int status = nvm_dma_map_host(&dma, ctrl, buffer, size);
    if (!nvm_ok(status))
    {
        cudaFreeHost(buffer);
        throw error(string("Failed to map host memory: ") + nvm_strerror(status));
    }

    return DmaPtr(dma, [buffer](nvm_dma_t* dma) {
        nvm_dma_unmap(dma);
        cudaFreeHost(buffer);
    });
}



DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice)
{
    if (cudaDevice < 0)
    {
        return createDma(ctrl, size);
    }

    nvm_dma_t* dma = nullptr;
    void* bufferPtr = nullptr;
    void* devicePtr = nullptr;

    getDeviceMemory(cudaDevice, bufferPtr, devicePtr, size);

    int status = nvm_dma_map_device(&dma, ctrl, devicePtr, size);
    if (!nvm_ok(status))
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to map device memory: ") + nvm_strerror(status));
    }

    //dma->vaddr = bufferPtr;

    return DmaPtr(dma, [bufferPtr](nvm_dma_t* dma) {
        nvm_dma_unmap(dma);
        cudaFree(bufferPtr);
    });
}



BufferPtr createBuffer(size_t size)
{
    void* buffer = nullptr;

    cudaError_t err = cudaHostAlloc(&buffer, size, cudaHostAllocDefault);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to allocate host memory: ") + cudaGetErrorString(err));
    }

    return BufferPtr(buffer, [](void* ptr) { cudaFreeHost(ptr); });
}



BufferPtr createBuffer(size_t size, int cudaDevice)
{
    if (cudaDevice < 0)
    {
        return createBuffer(size);
    }

    void* bufferPtr = nullptr;

    getDeviceMemory(cudaDevice, bufferPtr, size);

    return BufferPtr(bufferPtr, [](void* ptr) { cudaFree(ptr); });
}



#ifdef __DIS_CLUSTER__
DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, uint32_t, uint32_t)
{
    nvm_dma_t* dma = nullptr;

    int status = nvm_dis_dma_create(&dma, ctrl, size, 0);
    if (!nvm_ok(status))
    {
        throw error(string("Failed to create local segment: ") + nvm_strerror(status));
    }

    return DmaPtr(dma, nvm_dma_unmap);
}
#else
DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, uint32_t, uint32_t)
{
    return createDma(ctrl, size);
}
#endif


#ifdef __DIS_CLUSTER__
DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice, uint32_t, uint32_t)
{
    if (cudaDevice < 0)
    {
        return createDma(ctrl, size, 0, 0);
    }

    nvm_dma_t* dma = nullptr;
    void* bufferPtr = nullptr;
    void* devicePtr = nullptr;

    getDeviceMemory(cudaDevice, bufferPtr, devicePtr, size);

    int status = nvm_dis_dma_map_device(&dma, ctrl, devicePtr, size);
    if (!nvm_ok(status))
    {
        throw error(string("Failed to create local segment: ") + nvm_strerror(status));
    }

    //dma->vaddr = devicePtr;

    return DmaPtr(dma, nvm_dma_unmap);
}
#else
DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice, uint32_t, uint32_t)
{
    return createDma(ctrl, size, cudaDevice);
}
#endif



#ifdef __DIS_CLUSTER__
DmaPtr createRemoteDma(const nvm_ctrl_t* ctrl, size_t size)
{
    nvm_dma_t* dma = nullptr;

    int status = nvm_dis_dma_create(&dma, ctrl, size, SCI_MEMACCESS_HOST_WRITE | SCI_MEMACCESS_DEVICE_READ);
    if (!nvm_ok(status))
    {
        throw error(string("Failed to create device segment: ") + nvm_strerror(status));
    }

    // TODO: implement this
    //cudaError_t err = cudaHostRegister(

    return DmaPtr(dma, nvm_dma_unmap);
}
#endif

