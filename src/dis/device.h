#ifndef __NVM_INTERNAL_DIS_DEVICE_H__
#define __NVM_INTERNAL_DIS_DEVICE_H__
#ifdef _SISCI

/* Make sure everything is defined as needed */
#ifndef __DIS_CLUSTER__
#define __DIS_CLUSTER__
#endif

/* Necessary includes */
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <sisci_types.h>
#include "mutex.h"



/* 
 * Device descriptor.
 *
 * Holds a reference to a "borrowed" SISCI SmartIO device.
 */
struct device
{
    uint32_t                fdid;           // SISCI SmartIO device identifier (fabric device identifier)
    sci_desc_t              sd;             // SISCI virtual device descriptor
    struct mutex            lock;           // Ensure exclusive access to device
    uint32_t                counter;        // Segment identifier counter
    sci_smartio_device_t    device;         // SmartIO device handle
    sci_remote_segment_t    segment;        // Reference to PCI bar 0
    size_t                  size;           // Size of BAR0
    volatile void*          ptr;            // Mapped pointer
    sci_map_t               md;             // SISCI mapping descriptor
};



/*
 * Connect to SmartIO device memory data segment.
 */
int _nvm_device_memory_get(sci_remote_segment_t* segment, 
                           const struct device* dev, 
                           uint32_t id,
                           unsigned int memtype);



/*
 * Disconnect from SmartIO device memory segment.
 */
void _nvm_device_memory_put(sci_remote_segment_t* segment);



/*
 * Create local segment.
 * If ptr is not NULL, create empty segment and register segment memory.
 * If ptr is not NULL and gpu_mem is set, attach it CUDA device memory
 */
int _nvm_local_memory_get(sci_local_segment_t* segment,
                          uint32_t* adapter,
                          const struct device* dev,
                          size_t size,
                          void* ptr,
                          bool gpu_mem);



/*
 * Remove local segment.
 */
void _nvm_local_memory_put(sci_local_segment_t* segment);



#endif /* _SISCI */
#endif /* __NVM_INTERNAL_DIS_DEVICE_H__ */
