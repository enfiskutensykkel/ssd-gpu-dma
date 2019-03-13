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

using std::runtime_error;
using std::logic_error;
using std::string;



DmaPtr allocateBuffer(const nvm_ctrl_t* ctrl, size_t size, uint32_t, uint32_t)
{
    nvm_dma_t* dma = nullptr;

    auto err = nvm_dis_dma_create(&dma, ctrl, size, 0);
    if (!nvm_ok(err))
    {
        throw runtime_error(string("Failed to create local segment mapped for controller: ") + nvm_strerror(err));
    }

    return DmaPtr(dma, nvm_dma_unmap);
}



DmaPtr allocateBuffer(const Ctrl& ctrl, size_t size, uint32_t adapter, uint32_t segmentId)
{
    if (ctrl.fdid == 0)
    {
        return allocateBuffer(ctrl, size);
    }

    return allocateBuffer(ctrl.handle, size, adapter, segmentId);
}



DmaPtr allocateBuffer(const Ctrl& ctrl, size_t size, uint32_t segmentId)
{
    return allocateBuffer(ctrl, size, ctrl.adapter, segmentId);
}



DmaPtr deviceBuffer(const nvm_ctrl_t* ctrl, size_t size)
{
    nvm_dma_t* dma = nullptr;

    auto err = nvm_dis_dma_create(&dma, ctrl, size, SCI_MEMACCESS_HOST_WRITE | SCI_MEMACCESS_DEVICE_READ);
    if (!nvm_ok(err))
    {
        throw runtime_error(string("Failed to connect to remote segment: ") + nvm_strerror(err));
    }

    return DmaPtr(dma, nvm_dma_unmap);
}



DmaPtr deviceBuffer(const Ctrl& ctrl, size_t size)
{
    if (ctrl.fdid == 0)
    {
        throw logic_error("Controller is not a SmartIO device");
    }

    return deviceBuffer(ctrl.handle, size);
}

