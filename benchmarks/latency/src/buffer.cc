#include <nvm_types.h>
#include <nvm_dma.h>
#include <nvm_util.h>
#include <nvm_error.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "ctrl.h"
#include "buffer.h"

using error = std::runtime_error;
using std::string;



DmaPtr allocateBuffer(const nvm_ctrl_t* ctrl, size_t size)
{
    auto memory = allocateHostMemory(size);

    nvm_dma_t* dma = nullptr;
    int err = nvm_dma_map_host(&dma, ctrl, memory.get(), size);
    if (!nvm_ok(err))
    {
        throw error("Failed to map local RAM for controller: " + string(nvm_strerror(err)));
    }

    return DmaPtr(dma, [memory](nvm_dma_t* dma) mutable {
        nvm_dma_unmap(dma);
        memory.reset();
    });
}



#ifndef __DIS_CLUSTER__
DmaPtr allocateBuffer(const nvm_ctrl_t* ctrl, size_t size, uint32_t, uint32_t)
{
    return allocateBuffer(ctrl, size);
}
#endif



DmaPtr allocateBuffer(const Ctrl& ctrl, size_t size)
{
    return allocateBuffer(ctrl.handle, size);
}



#ifndef __DIS_CLUSTER__
DmaPtr allocateBuffer(const Ctrl& ctrl, size_t size, uint32_t, uint32_t)
{
    return allocateBuffer(ctrl.handle, size);
}
#endif



#ifndef __DIS_CLUSTER__
DmaPtr allocateBuffer(const Ctrl& ctrl, size_t size, uint32_t)
{
    return allocateBuffer(ctrl.handle, size);
}
#endif



MemoryBuffer::MemoryBuffer(const Ctrl& ctrl, size_t size)
    : size(size)
    , buffer(allocateBuffer(ctrl, size))
{
}



MemoryBuffer::MemoryBuffer(const Ctrl& ctrl, size_t size, uint32_t segmentId)
    : size(size)
    , buffer(allocateBuffer(ctrl, size, segmentId))
{
}



size_t MemoryBuffer::load(const void* src, size_t size, size_t offset)
{
    if (offset + size > this->size)
    {
        throw error("Specified offset and size is too large");
    }

    void* dst = (void*) (((unsigned char*) buffer->vaddr) + offset);
    memcpy(dst, src, size);
    return size;
}



size_t MemoryBuffer::save(void* dst, size_t size, size_t offset) const
{
    if (offset + size > this->size)
    {
        throw error("Specified offset and size is too large");
    }

    void* src = (void*) (((unsigned char*) buffer->vaddr) + offset);
    memcpy(dst, src, size);
    return size;
}



void MemoryBuffer::clear()
{
    memset(buffer->vaddr, 0, size);
}

