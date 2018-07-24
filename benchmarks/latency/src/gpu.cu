#include <cuda.h>
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
#include <sstream>
#include <iomanip>
#include <cstddef>
#include <cstdint>

using error = std::runtime_error;
using std::string;

#define GPU_PAGE_SIZE 0x10000


Gpu::Gpu(int device)
    : device(nullptr)
    , cudaDevice(device)
    , fdid(0)
    , adapter(0)
{
    // Some sanity checking
    setDevice();
}



void Gpu::setDevice() const
{
    cudaError_t err = cudaSetDevice(cudaDevice);
    if (err != cudaSuccess)
    {
        throw error("Failed to set CUDA device: " + string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        throw error("Failed to synchronize device: " + string(cudaGetErrorString(err)));
    }
}



string Gpu::deviceBdf() const
{
    using namespace std;

    cudaDeviceProp properties;

    cudaError_t err = cudaGetDeviceProperties(&properties, cudaDevice);
    if (err != cudaSuccess)
    {
        throw error("Failed to get device properties: " + string(cudaGetErrorString(err)));
    }

    ostringstream s;
    s << setfill('0') << setw(4) << hex << properties.pciDomainID
        << ":" << setfill('0') << setw(2) << hex << properties.pciBusID
        << ":" << setfill('0') << setw(2) << hex << properties.pciDeviceID
        << ".0";

    return s.str();
}



string Gpu::deviceName() const
{
    cudaDeviceProp properties;

    cudaError_t err = cudaGetDeviceProperties(&properties, cudaDevice);
    if (err != cudaSuccess)
    {
        throw error("Failed to get device properties: " + string(cudaGetErrorString(err)));
    }
    
    return properties.name;
}



void* Gpu::getDevicePointer(const MemoryPtr& memory)
{
    void* pointer = memory.get();
    if (pointer == nullptr)
    {
        return nullptr;
    }

    cudaPointerAttributes attrs;
    auto err = cudaPointerGetAttributes(&attrs, pointer);
    if (err != cudaSuccess)
    {
        throw error("Failed to get pointer attributes: " + string(cudaGetErrorString(err)));
    }

    return attrs.devicePointer;
}



MemoryPtr Gpu::allocateMemory(size_t size) const
{
    setDevice();

    size = NVM_PAGE_ALIGN(size, GPU_PAGE_SIZE); // FIXME: Hack

    void* pointer = nullptr;
    auto err = cudaMalloc(&pointer, size);
    if (err != cudaSuccess)
    {
        throw error("Failed to allocate device memory: " + string(cudaGetErrorString(err)));
    }

    err = cudaMemset(pointer, 0, size);
    if (err != cudaSuccess)
    {
        cudaFree(pointer);
        throw error("Failed to clear device memory: " + string(cudaGetErrorString(err)));
    }

    return MemoryPtr(pointer, cudaFree);
}



DmaPtr Gpu::allocateBuffer(const Ctrl& controller, size_t size) const
{
    auto memory = allocateMemory(size);
    void* pointer = getDevicePointer(memory);

    nvm_dma_t* dma = nullptr;
    int status = nvm_dma_map_device(&dma, controller.handle, pointer, size);
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



#ifndef __DIS_CLUSTER__
DmaPtr Gpu::allocateBuffer(const Ctrl& controller, size_t size, uint32_t) const
{
    return allocateBuffer(controller, size);
}
#endif



int Gpu::findDevice(int domain, int bus, int devfn)
{
    int max = deviceCount();

    for (int i = 0; i < max; ++i)
    {
        cudaDeviceProp properties;

        auto err = cudaGetDeviceProperties(&properties, i);
        if (err != cudaSuccess)
        {
            throw error("Failed to get device properties: " + string(cudaGetErrorString(err)));
        }

        if (domain == properties.pciDomainID && bus == properties.pciBusID && devfn == properties.pciDeviceID)
        {
            return i;
        }
    }

    throw error("Specified device is not a CUDA device!");
}



int Gpu::deviceCount()
{
    int max = 0;

    auto err = cudaGetDeviceCount(&max);
    if (err != cudaSuccess)
    {
        throw error("Failed to get number of devices: " + string(cudaGetErrorString(err)));
    }

    return max;
}



#ifndef __DIS_CLUSTER__
Gpu::Gpu(uint64_t, uint32_t)
    : cudaDevice(-1)
    , fdid(0)
    , adapter(0)
{
    throw std::logic_error("SmartIO not supported");
}
#endif



GpuBuffer::GpuBuffer(const GpuPtr& gpu, const Ctrl& ctrl, size_t size)
    : MemoryBuffer(gpu->allocateBuffer(ctrl, size), size)
    , gpu(gpu)
{
}



GpuBuffer::GpuBuffer(const GpuPtr& gpu, const Ctrl& ctrl, size_t size, uint32_t segmentId)
    : MemoryBuffer(gpu->allocateBuffer(ctrl, size, segmentId), size)
    , gpu(gpu)
{
}



size_t GpuBuffer::load(const void* src, size_t size, size_t offset)
{
    if (offset + size > this->size)
    {
        throw error("Specified offset and size is too large");
    }

    void* dst = (void*) (((unsigned char*) buffer->vaddr) + offset);

    gpu->setDevice();

    auto err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        throw error("Failed to copy to device memory: " + string(cudaGetErrorString(err)));
    }

    return size;
}



size_t GpuBuffer::save(void* dst, size_t size, size_t offset) const
{
    if (offset + size > this->size)
    {
        throw error("Specified offset and size is too large");
    }

    void* src = (void*) (((unsigned char*) buffer->vaddr) + offset);

    gpu->setDevice();

    auto err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        throw error("Failed to copy to host memory: " + string(cudaGetErrorString(err)));
    }

    return size;
}



void GpuBuffer::clear()
{
    gpu->setDevice();

    auto err = cudaMemset(buffer->vaddr, 0, size);
    if (err != cudaSuccess)
    {
        throw error("Failed to clear device memory: " + string(cudaGetErrorString(err)));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        throw error("Failed to synchronize device: " + string(cudaGetErrorString(err)));
    }
}

