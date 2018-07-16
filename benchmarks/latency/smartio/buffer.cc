#include <nvm_types.h>
#include <nvm_dma.h>
#include <nvm_util.h>
#include <nvm_error.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <cstddef>
#include <cstdint>
#include "ctrl.h"
#include "buffer.h"

using error = std::runtime_error;
using std::string;



DmaPtr allocateBuffer(const nvm_ctrl_t* ctrl, size_t size, uint32_t adapter, uint32_t segmentId)
{
    nvm_dma_t* dma = nullptr;

    auto err = nvm_dis_dma_create(&dma, ctrl, adapter, segmentId, size);
    if (!nvm_ok(err))
    {
        throw error(string("Failed to create local segment mapped for controller: ") + nvm_strerror(err));
    }

    return DmaPtr(dma, nvm_dma_unmap);
}



DmaPtr allocateBuffer(const Ctrl& ctrl, size_t size, uint32_t adapter, uint32_t segmentId)
{
    return allocateBuffer(ctrl.handle, size, adapter, segmentId);
}



DmaPtr allocateBuffer(const Ctrl& ctrl, size_t size, uint32_t segmentId)
{
    return allocateBuffer(ctrl.handle, size, ctrl.adapter, segmentId);
}



DmaPtr connectBuffer(const nvm_ctrl_t* ctrl, size_t size, uint32_t adapter, uint32_t number)
{
    nvm_dma_t* dma = nullptr;

    auto err = nvm_dis_dma_connect(&dma, ctrl, adapter, number, size, true);
    if (!nvm_ok(err))
    {
        throw error(string("Failed to connect to remote segment: ") + nvm_strerror(err));
    }

    return DmaPtr(dma, nvm_dma_unmap);
}



DmaPtr connectBuffer(const Ctrl& ctrl, size_t size, uint32_t adapter, uint32_t number)
{
    return connectBuffer(ctrl.handle, size, adapter, number);
}

