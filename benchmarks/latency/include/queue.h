#ifndef __LATENCY_BENCHMARK_QUEUE_H__
#define __LATENCY_BENCHMARK_QUEUE_H__

#include <nvm_types.h>
#include "ctrl.h"
#include "buffer.h"
#include "gpu.h"
#include <memory>
#include <map>
#include <string>
#include <cstdint>
#include <cstddef>


enum QueueLocation : int
{
    REMOTE, LOCAL, GPU
};


/*
 * Container for a queue pair (CQ + SQ).
 */
struct QueuePair
{
    public:
        const uint16_t no;  // Queue number
        const size_t depth; // Queue depth (number of commands)
        const size_t pages; // Maximum data transfer size (in pages)

        mutable nvm_queue_t sq;
        mutable nvm_queue_t cq;


        QueuePair(const CtrlPtr& controller, uint16_t no, size_t depth, size_t pages, DmaPtr queueMemory);


        QueuePair(const CtrlPtr& controller, uint16_t no, size_t depth, size_t pages, DmaPtr cqMemory, DmaPtr sqMemory);

        virtual ~QueuePair();

        virtual std::string type() const = 0;

        virtual QueueLocation location() const = 0;

        const Ctrl& getController() const
        {
            return *controller;
        }

        const DmaPtr& getQueueMemory() const
        {
            return sqMemory;
        }

    protected:
        QueuePair();

    private:
        CtrlPtr controller;
        DmaPtr cqMemory;
        DmaPtr sqMemory;
};


typedef std::shared_ptr<QueuePair> QueuePtr;
typedef std::map<uint16_t, QueuePtr> QueueMap;



struct GpuQueue : public QueuePair
{
    public:
        GpuQueue(const CtrlPtr& controller, 
                 uint16_t no, 
                 size_t depth, 
                 size_t pages,
                 const GpuPtr& gpu, 
                 uint32_t adapter, 
                 uint32_t cqSegmentId, 
                 uint32_t sqSegmentId);

        std::string type() const override;

        QueueLocation location() const override
        {
            return QueueLocation::GPU;
        }

        const Gpu& getGpu() const
        {
            return *gpu;
        }

    private:
        GpuPtr gpu;
};



struct LocalQueue : public QueuePair
{
    public:
        std::string type() const override;

        LocalQueue(const CtrlPtr& controller, uint16_t no, size_t depth, size_t pages);

        LocalQueue(const CtrlPtr& controller, uint16_t no, size_t depth, size_t pages, uint32_t adapter, uint32_t segmentId);

        QueueLocation location() const override
        {
            return QueueLocation::LOCAL;
        }
};



struct RemoteQueue : public QueuePair
{
    public:
        std::string type() const override;

        RemoteQueue(const CtrlPtr& controller, uint16_t no, size_t depth, size_t pages, uint32_t adapter, uint32_t segmentId);

        QueueLocation location() const override
        {
            return QueueLocation::REMOTE;
        }
};


#endif /* __LATENCY_BENCHMARK_QUEUE_H__ */
