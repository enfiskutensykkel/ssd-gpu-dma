#include <cuda.h>
#include <nvm_util.h>
#include <nvm_types.h>
#include <nvm_dma.h>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include "dma.h"

using std::runtime_error;

static const size_t GPU_BOUND_SIZE = 0x10000;


__host__ static void deleteHandle(nvm_dma_t* handle)
{
    nvm_dma_window_free(*handle);
    delete handle;
}


__host__ DmaPtr createHostBuffer(nvm_ctrl_t ctrl, size_t size)
{
    void* memoryPtr = nullptr;

    int err = posix_memalign(&memoryPtr, ctrl->page_size, size);
    if (memoryPtr == nullptr || err != 0)
    {
        throw runtime_error("Failed to allocate host buffer");
    }

    nvm_dma_t* dma_window = new (std::nothrow) nvm_dma_t;
    if (dma_window == nullptr)
    {
        free(memoryPtr);
        throw runtime_error("Failed to allocate DMA mapping handle");
    }

    err = nvm_dma_window_host_map(dma_window, ctrl, memoryPtr, size);
    if (err != 0)
    {
        delete dma_window;
        free(memoryPtr);
        throw runtime_error("Failed to map host memory for DMA (" + std::string(strerror(err)) + ")");
    }
    memset(memoryPtr, 0, size);

    return DmaPtr(dma_window, [memoryPtr](nvm_dma_t* handle) {
        deleteHandle(handle);
        free(memoryPtr);
    });
}


__host__ DmaPtr createDeviceBuffer(nvm_ctrl_t ctrl, size_t size, int cudaDevice)
{
    void* memoryPtr = nullptr;
    size = DMA_SIZE(size, GPU_BOUND_SIZE);

    cudaError_t err = cudaSetDevice(cudaDevice);
    if (err != cudaSuccess)
    {
        throw runtime_error("Failed to set CUDA device");
    }

    err = cudaMalloc(&memoryPtr, size);
    if (err != cudaSuccess)
    {
        throw runtime_error("Failed to allocate device buffer");
    }

    cudaPointerAttributes attrs;
    err = cudaPointerGetAttributes(&attrs, memoryPtr);
    if (err != cudaSuccess)
    {
        cudaFree(memoryPtr);
        throw runtime_error("Failed to get pointer attributes");
    }
    void* devicePtr = attrs.devicePointer;

    nvm_dma_t* dma_window = new (std::nothrow) nvm_dma_t;
    if (dma_window == nullptr)
    {
        cudaFree(memoryPtr);
        throw runtime_error("Failed to allocate DMA mapping handle");
    }

    int nvmerr = nvm_dma_window_device_map(dma_window, ctrl, devicePtr, size);
    if (nvmerr != 0)
    {
        delete dma_window;
        cudaFree(memoryPtr);
        throw runtime_error("Failed to map device memory for DMA");
    }
    cudaMemset(memoryPtr, 0, size);

    // Ugly bugly hack, but it works...
    (*dma_window)->vaddr = memoryPtr;

    return DmaPtr(dma_window, [memoryPtr, devicePtr](nvm_dma_t* handle) {
        (*handle)->vaddr = devicePtr;
        deleteHandle(handle);
        cudaFree(memoryPtr);
    });
}

