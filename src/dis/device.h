#ifndef __NVM_INTERNAL_DIS_DEVICE_H__
#define __NVM_INTERNAL_DIS_DEVICE_H__

/* Forward declarations */
struct device;
struct device_memory;


#ifdef _SISCI

/* Make sure everything is defined as needed */
#ifndef __DIS_CLUSTER__
#define __DIS_CLUSTER__
#endif

/* Necessary includes */
#include <stddef.h>
#include <stdint.h>
#include <sisci_types.h>
#include "dis/map.h"



/* 
 * Device descriptor.
 *
 * Holds a "borrowed" reference to a SISCI SmartIO device.
 * Handles that require a device or controller reference should take this.
 */
struct device
{
    sci_desc_t              sd;             // SISCI virtual device descriptor
    uint64_t                device_id;      // SISCI SmartIO device identifier
    sci_device_t            device;         // SmartIO device handle
};



/*
 * Device memory segment descriptor.
 *
 * Describes mapping to device-local memory, e.g. PCI BARs or memory residing 
 * on the same host as the device.
 */
struct device_memory
{
    struct device           device;         // Device reference
    uint32_t                adapter;        // DIS adapter number
    uint32_t                segment_no;     // Device segment number
    sci_remote_segment_t    segment;        // SISCI remote segment to device memory
    uint32_t                flags;          // SISCI flags used when connecting segment
};



/*
 * Acquire device reference (increase).
 */
int _nvm_device_get(struct device* dev, uint64_t dev_id);



/*
 * Release device reference (decrease).
 */
void _nvm_device_put(struct device* dev);



/*
 * Connect to device memory.
 * This function will increase the device reference.
 */
int _nvm_device_memory_get(struct device_memory* mem, 
                           const struct device* dev, 
                           uint32_t adapter,
                           uint32_t segment_no,
                           uint32_t flags);



/*
 * Disconnect from device memory.
 * This function will decrease the device reference.
 */
void _nvm_device_memory_put(struct device_memory* mem);



#endif /* _SISCI */
#endif /* __NVM_INTERNAL_DIS_DEVICE_H__ */
