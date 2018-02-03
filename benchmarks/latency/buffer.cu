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
#include <cstdio>

using error = std::runtime_error;
using std::string;



BufferPtr createBuffer(const nvm_ctrl_t* ctrl, uint32_t adapter, uint32_t id, size_t size, int dev)
{
    nvm_dma_t* dma = nullptr;
    void* bufferPtr = nullptr;

    if (dev < 0)
    {
        return createBuffer(ctrl, adapter, id, size);
    }

    cudaError_t err = cudaSetDevice(dev);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to set CUDA device: ") + cudaGetErrorString(err));
    }

    err = cudaMalloc(&bufferPtr, size);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to allocate device memory: ") + cudaGetErrorString(err));
    }

    cudaPointerAttributes attrs;
    err = cudaPointerGetAttributes(&attrs, bufferPtr);
    if (err != cudaSuccess)
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to get pointer attributes: ") + cudaGetErrorString(err));
    }

    fprintf(stderr, "bufferPtr=%p devicePointer=%p\n", bufferPtr, attrs.devicePointer);

#ifdef __DIS_CLUSTER__
    int status = nvm_dis_dma_map_device(&dma, ctrl, adapter, id, attrs.devicePointer, size);
#else
    int status = nvm_dma_map_device(&dma, ctrl, attrs.devicePointer, size);
#endif
    if (!nvm_ok(status))
    {
        cudaFree(bufferPtr);
        throw error(string("Failed to create local segment: ") + nvm_strerror(status));
    }

    dma->vaddr = bufferPtr;

    return BufferPtr(dma, [bufferPtr](nvm_dma_t* m) {
        nvm_dma_unmap(m);
        if (bufferPtr != nullptr)
        {
            cudaFree(bufferPtr);
        }
    });
}



#ifdef __DIS_CLUSTER__
BufferPtr createBuffer(const nvm_ctrl_t* ctrl, uint32_t adapter, uint32_t id, size_t size)
{
    nvm_dma_t* dma = nullptr;

    int status = nvm_dis_dma_create(&dma, ctrl, adapter, id, size);
    if (!nvm_ok(status))
    {
        throw error(string("Failed to create local segment: ") + nvm_strerror(status));
    }

    return BufferPtr(dma, [](nvm_dma_t* m) { nvm_dma_unmap(m); });
}
#else
BufferPtr createBuffer(const nvm_ctrl_t* ctrl, uint32_t, uint32_t, size_t size)
{
    nvm_dma_t* dma = nullptr;
    void* ptr = nullptr;

    int status = posix_memalign(&ptr, ctrl->page_size, size);
    if (status != 0)
    {
        throw error(string("Failed to allocate memory: ") + strerror(errno));
    }

    status = nvm_dma_map_host(&dma, ctrl, ptr, size);
    if (!nvm_ok(status))
    {
        free(ptr);
        throw error(string("Failed to map host memory: ") + nvm_strerror(status));
    }

    return BufferPtr(dma, [ptr](nvm_dma_t* m) {
        nvm_dma_unmap(m);
        if (ptr != nullptr)
        {
            free(ptr);
        }
    });
}
#endif



#ifdef __DIS_CLUSTER__
BufferPtr createRemoteBuffer(const nvm_ctrl_t* ctrl, uint32_t adapter, uint32_t segno, size_t size)
{
    nvm_dma_t* dma = nullptr;

    int status = nvm_dis_dma_connect(&dma, ctrl, adapter, segno, size, true); // FIXME: should be private
    if (!nvm_ok(status))
    {
        throw error(string("Failed to connect to segment: ") + nvm_strerror(status));
    }

    return BufferPtr(dma, [](nvm_dma_t* m) { nvm_dma_unmap(m); });
}
#endif

