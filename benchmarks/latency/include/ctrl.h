#ifndef __LATENCY_BENCHMARK_CTRL_H__
#define __LATENCY_BENCHMARK_CTRL_H__

#include <nvm_types.h>
#include "ctrl.h"
#include "buffer.h"
#include <memory>
#include <string>
#include <cstddef>
#include <cstdint>


/*
 * NVMe controller reference wrapper.
 */
struct Ctrl final
{
    private:
        mutable nvm_ctrl_t* writeHandle;
        mutable DmaPtr      adminQueueMemory;
        mutable nvm_dma_t*  adminQueues;
        mutable nvm_aq_ref  adminRef;

    public:
        const nvm_ctrl_t*   handle;         /* Read only-handle */
        const uint64_t      fdid;           /* SmartIO fabric device identifier */
        const uint32_t      adapter;        /* SmartIO NTB adapter */
        uint32_t            namespaceId;    /* NVM namespace identifier */
        uint64_t            namespaceSize;  /* Size of namespace in blocks */
        size_t              pageSize;       /* Controller page size */
        size_t              blockSize;      /* Block size */
        size_t              chunkSize;      /* Maximum data transfer size */
        uint16_t            numQueues;      /* Maximum number of queue pairs */
        size_t              maxEntries;

        ~Ctrl() noexcept;

        nvm_aq_ref adminReference() const
        {
            return adminRef;
        }

    private:
        friend struct CtrlManager;
        Ctrl(const Ctrl&) = delete;
        Ctrl(Ctrl&&) noexcept = delete;
        Ctrl& operator=(const Ctrl&) = delete;
        Ctrl& operator=(const Ctrl&&) noexcept = delete;
        Ctrl(uint64_t fdid, uint32_t adapter, nvm_ctrl_t* controller, DmaPtr memory, nvm_dma_t* dma, nvm_aq_ref ref, uint32_t ns);
};



/*
 * Declare controller reference pointer type.
 */
typedef std::shared_ptr<const Ctrl> CtrlPtr;



/*
 * NVMe controller reference manager.
 */
struct CtrlManager final
{
    public:
        CtrlManager(const std::string& devicePath, uint32_t nvmNamespace);


        CtrlManager(uint64_t fdid, uint32_t adapter, uint32_t segmentId, bool adminManager, uint32_t nvmNamespace);


        ~CtrlManager();


        const Ctrl& getController() const
        {
            return *controller;
        }

        const CtrlPtr& getControllerPtr() const
        {
            return controller;
        }

    private:
        CtrlPtr     controller;
        int         fileDescriptor;
};


/* Convenience type */
typedef std::shared_ptr<CtrlManager> CtrlManagerPtr;

#endif /* __LATENCY_BENCHMARK_CTRL_H__ */
