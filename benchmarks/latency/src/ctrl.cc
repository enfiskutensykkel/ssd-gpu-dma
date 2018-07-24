#include "ctrl.h"
#include "buffer.h"
#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_aq.h>
#include <nvm_admin.h>
#include <nvm_dma.h>
#include <nvm_util.h>
#include <nvm_error.h>
#include <nvm_rpc.h>
#include <string>
#include <memory>
#include <new>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>
#include <cerrno>

using error = std::runtime_error;
using std::string;



Ctrl::Ctrl(uint64_t fdid, uint32_t adapter, nvm_ctrl_t* controller, DmaPtr memory, nvm_dma_t* dma, nvm_aq_ref ref, uint32_t ns)
    : writeHandle(controller)
    , adminQueueMemory(memory)
    , adminQueues(dma)
    , adminRef(ref)
    , handle(controller)
    , fdid(fdid)
    , adapter(adapter)
    , namespaceId(ns)
{
    // Identify cntroller
    nvm_ctrl_info ctrlInfo;
    int status = nvm_admin_ctrl_info(adminRef, &ctrlInfo, NVM_DMA_OFFSET(adminQueues, 2), adminQueues->ioaddrs[2]);
    if (!nvm_ok(status))
    {
        throw error("Failed to identify controller: " + string(nvm_strerror(status)));
    }

    // Identify namespace
    nvm_ns_info nsInfo;
    status = nvm_admin_ns_info(adminRef, &nsInfo, ns, NVM_DMA_OFFSET(adminQueues, 2), adminQueues->ioaddrs[2]);
    if (!nvm_ok(status))
    {
        throw error("Failed to identify namespace: " + string(nvm_strerror(status)));
    }

    // Figure out the maximum number of queues
    uint16_t numSQs = 0;
    uint16_t numCQs = 0;
    status = nvm_admin_get_num_queues(adminRef, &numCQs, &numSQs);
    if (!nvm_ok(status))
    {
        throw error("Failed to get number of queues: " + string(nvm_strerror(status)));
    }

    // Set members
    namespaceSize = nsInfo.size;
    pageSize = ctrlInfo.page_size;
    blockSize = nsInfo.lba_data_size;
    chunkSize = ctrlInfo.max_data_size;
    numQueues = std::min(numSQs, numCQs);
    maxEntries = std::min((size_t) ctrlInfo.max_entries, ctrlInfo.page_size / ctrlInfo.sq_entry_size);
}



Ctrl::~Ctrl()
{
    nvm_aq_destroy(adminRef);
    if (adminQueueMemory.get() == nullptr)
    {
        nvm_dma_unmap(adminQueues);
    }
    adminQueueMemory.reset();
    nvm_ctrl_free(writeHandle);
}



CtrlManager::CtrlManager(const string& path, uint32_t ns)
    : controller(nullptr)
    , fileDescriptor(-1)
{
    // Open file descriptor
    fileDescriptor = open(path.c_str(), O_RDWR | O_NONBLOCK);
    if (fileDescriptor < 0)
    {
        throw error("Failed to open file descriptor: " + string(strerror(errno)));
    }

    DmaPtr queueMemory;
    nvm_ctrl_t* ctrl = nullptr;
    nvm_aq_ref ref = nullptr;

    try
    {
        // Get controller reference
        int status = nvm_ctrl_init(&ctrl, fileDescriptor);
        if (!nvm_ok(status))
        {
            throw error("Failed to get controller reference: " + string(nvm_strerror(status)));
        }

        // Map queue memory for controller
        queueMemory = allocateBuffer(ctrl, ctrl->page_size * 3);

        // Create admin queue reference
        status = nvm_aq_create(&ref, ctrl, queueMemory.get());
        if (!nvm_ok(status))
        {
            throw error("Failed to reset controller: " + string(nvm_strerror(status)));
        }

        // Create controller reference wrapper
        auto* ptr = new (std::nothrow) Ctrl(0, 0, ctrl, queueMemory, queueMemory.get(), ref, ns);
        if (ptr == nullptr)
        {
            throw error("Failed to allocate shared controller reference");
        }

        controller.reset(ptr); 
    }
    catch (const error& e)
    {
        nvm_aq_destroy(ref);
        queueMemory.reset();
        nvm_ctrl_free(ctrl);
        close(fileDescriptor);
        throw e;
    }
}


CtrlManager::~CtrlManager()
{
    controller.reset();
    if (fileDescriptor >= 0)
    {
        close(fileDescriptor);
    }
}



#ifndef __DIS_CLUSTER__
CtrlManager::CtrlManager(uint64_t, uint32_t, uint32_t, bool, uint32_t)
    : controller(nullptr)
    , fileDescriptor(-1)
{
    throw std::logic_error("SmartIO unsupported");
}
#endif

