#ifndef __LATENCY_BENCHMARK_BUFFER_H__
#define __LATENCY_BENCHMARK_BUFFER_H__

#include <memory>
#include <cstddef>
#include <cstdint>
#include <nvm_types.h>

/* Forward declaration */
struct Ctrl;



/*
 * Convenience type for DMA handles.
 * Will automaticall clean itself up once not used anymore.
 */
typedef std::shared_ptr<nvm_dma_t> DmaPtr;



/*
 * Convenience type for a (non-mapped) memory buffer.
 */
typedef std::shared_ptr<void> MemoryPtr;



/*
 * Memory buffer type.
 */
struct MemoryBuffer
{
    public:
        const size_t size;
        const DmaPtr buffer;
    

        MemoryBuffer(const Ctrl& controller, size_t size);


        MemoryBuffer(const Ctrl& controller, size_t size, uint32_t segmentId);


        virtual ~MemoryBuffer() {}


        /*
         * Copy memory from ptr and store in buffer.
         */
        virtual size_t load(const void* ptr, size_t size, size_t offset = 0);


        
        /*
         * Copy memory from buffer and store in ptr.
         */
        virtual size_t save(void* ptr, size_t size, size_t offset = 0) const;


        /*
         * Set memory to zero.
         */
        virtual void clear();

    protected:
        MemoryBuffer(const DmaPtr& buffer, size_t size)
            : size(size), buffer(buffer) {}
};



/* Convenience type */
typedef std::shared_ptr<MemoryBuffer> MemoryBufferPtr;



/*
 * Allocate memory in local RAM mapped for the NVMe controller.
 */
DmaPtr allocateBuffer(const nvm_ctrl_t* controller, size_t size);



/*
 * Allocate memory in local RAM mapped for the NVMe controller.
 */
DmaPtr allocateBuffer(const Ctrl& controller, size_t size);



/*
 * Allocate memory in local RAM.
 * This memory is not mapped for the NVMe controller.
 */
MemoryPtr allocateHostMemory(size_t size);



/*
 * Allocate memory in local RAM mapped for the NVMe controller.
 */
DmaPtr allocateBuffer(const nvm_ctrl_t* controller, size_t size, uint32_t adapter, uint32_t segmentId);



/*
 * Allocate memory in local RAM mapped for the NVMe controller.
 */
DmaPtr allocateBuffer(const Ctrl& controller, size_t size, uint32_t adapter, uint32_t segmentId);


/*
 * Allocate memory in local RAM mapped for the NVMe controller.
 */
DmaPtr allocateBuffer(const Ctrl& controller, size_t size, uint32_t segmentId);



#ifdef __DIS_CLUSTER__
DmaPtr connectBuffer(const nvm_ctrl_t* controller, size_t size, uint32_t adapter, uint32_t remoteSegmentId);
DmaPtr connectBuffer(const Ctrl& controller, size_t size, uint32_t adapter, uint32_t remoteSegmentId);
#endif


#endif /* __LATENCY_BENCHMARK_BUFFER_H__ */
