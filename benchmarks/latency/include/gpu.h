#ifndef __LATENCY_BENCHMARK_GPU_H__
#define __LATENCY_BENCHMARK_GPU_H__

#include "buffer.h"
#include <nvm_types.h>
#include <memory>
#include <string>
#include <cstddef>

/* Forward declaration */
struct Device;
struct Ctrl;


/*
 * Wrapper for GPU functionality, such as allocating and mapping
 * memory buffers for an NVMe controller.
 */
struct Gpu final
{
    private:
        std::shared_ptr<Device> device; /* SmartIO device wrapper */

    public:
        const int       cudaDevice; /* CUDA device ID */
        const uint64_t  fdid;       /* Fabric device ID (used by SmartIO) */
        const uint32_t  adapter;    /* NTB adapter number where GPU can be reached */


        /* Get a string representation of the BDF */
        std::string deviceBdf() const;


        /* Get canonical GPU name */
        std::string deviceName() const;

    
        /*
         * Set current GPU for CUDA context.
         */
        void setDevice() const;


        /*
         * Get number of CUDA devices
         */
        static int deviceCount();


        /*
         * Initialize instance using specified CUDA device.
         */
        Gpu(int cudaDevice);


        /*
         * Initialize instance using specified fabric device ID.
         */
        Gpu(uint64_t fdid, uint32_t adapter);


#ifdef __DIS_CLUSTER__
        /*
         * Allocate buffer in GPU memory and map it for the NVMe controller,
         * as well as the local CPU.
         * This is only possible for remote GPUs using SmartIO.
         */
        DmaPtr allocateBufferAndMap(const Ctrl& controller, size_t size, uint32_t segmentId) const;
#endif

    
        /*
         * Allocate buffer in GPU memory.
         * This memory is not mapped for the NVMe controller.
         */
        MemoryPtr allocateMemory(size_t size) const;


        /*
         * Allocate buffer in GPU memory and map it for the NVMe controller.
         */
        DmaPtr allocateBuffer(const Ctrl& controller, size_t size) const;


        /*
         * Allocate buffer in GPU memory and map it for the NVMe controller using SamrtIO.
         */
        DmaPtr allocateBuffer(const Ctrl& controller, size_t size, uint32_t segmentId) const;



        /*
         * Helper function to look up a device from its BDF.
         */
        static int findDevice(int domain, int bus, int devfn);


    private:
        /*
         * Helper function to look up a device pointer from a CUDA pointer.
         */
        static void* getDevicePointer(const MemoryPtr&);
};




/*
 * Convenience type.
 */
typedef std::shared_ptr<Gpu> GpuPtr;




struct GpuBuffer : public MemoryBuffer
{
    public:
        const GpuPtr gpu;

        GpuBuffer(const GpuPtr& gpu, const Ctrl& controller, size_t size);


        GpuBuffer(const GpuPtr& gpu, const Ctrl& controller, size_t size, uint32_t segmentId);


        size_t load(const void* ptr, size_t size, size_t offset = 0) override;


        size_t save(void* ptr, size_t size, size_t offset = 0) const override;


        void clear() override;
};


#endif /* __LATENCY_BENCHMARK_GPU_H__ */
