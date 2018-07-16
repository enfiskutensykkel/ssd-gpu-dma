#include "ctrl.h"
#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_dma.h>
#include <nvm_admin.h>
#include <nvm_util.h>
#include <nvm_rpc.h>
#include <nvm_aq.h>
#include <nvm_error.h>
#include <nvm_rpc.h>
#include <string>
#include <new>
#include <memory>
#include <stdexcept>
#include <cstdint>
#include <algorithm>

using error = std::runtime_error;
using std::string;



CtrlManager::CtrlManager(uint64_t fdid, uint32_t adapter, uint32_t segmentId, bool admin, uint32_t ns)
    : controller(nullptr)
    , fileDescriptor(-1)
{
    nvm_ctrl_t* ctrl = nullptr;
    nvm_dma_t* dma = nullptr;
    nvm_aq_ref ref = nullptr;

    try
    {
        // Get controller reference
        int status = nvm_dis_ctrl_init(&ctrl, fdid, adapter);
        if (!nvm_ok(status))
        {
            throw error("Failed to get controller reference: " + string(nvm_strerror(status)));
        }

        // Create and map queue memory for controller
        status = nvm_dis_dma_create(&dma, ctrl, adapter, segmentId, 3 * ctrl->page_size);
        if (!nvm_ok(status))
        {
            throw error("Failed to create admin queue memory: " + string(nvm_strerror(status)));
        }

        // Create or retrieve admin queue reference
        if (admin)
        {
            status = nvm_aq_create(&ref, ctrl, dma);
        }
        else
        {
            status = nvm_dis_rpc_bind(&ref, ctrl, adapter);
        }
        
        if (!nvm_ok(status))
        {
            throw error("Failed to initialize admin queues: " + string(nvm_strerror(status)));
        }

        // Create controller reference wrapper
        auto* ptr = new (std::nothrow) Ctrl(fdid, adapter, ctrl, nullptr, dma, ref, ns);
        if (ptr == nullptr)
        {
            throw error("Failed to allocate shared controller reference");
        }

        controller.reset(ptr); 
    } 
    catch (const error& e)
    {
        nvm_aq_destroy(ref);
        nvm_dma_unmap(dma);
        nvm_ctrl_free(ctrl);
        throw e;
    }
}

