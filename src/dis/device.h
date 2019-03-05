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


/* 
 * Device descriptor.
 *
 * Holds a reference to a "borrowed" SISCI SmartIO device.
 */
struct device
{
    uint32_t                fdid;           // SISCI SmartIO device identifier (fabric device identifier)
    sci_desc_t              sd;             // SISCI virtual device descriptor
    sci_smartio_device_t    device;         // SmartIO device handle
    sci_remote_segment_t    segment;        // Reference to PCI bar 0
    size_t                  size;           // Size of BAR0
    volatile void*          ptr;            // Mapped pointer
    sci_map_t               md;             // SISCI mapping descriptor
};



/*
 * Connect to SmartIO device memory data segment.
 */
int _nvm_connect_device_memory(sci_remote_segment_t* segment, 
                               const struct device* dev, 
                               uint32_t id,
                               unsigned int memtype);



/*
 * Disconnect from SmartIO device memory segment.
 */
void _nvm_disconnect_device_memory(sci_remote_segment_t* segment);


#endif /* _SISCI */
#endif /* __NVM_INTERNAL_DIS_DEVICE_H__ */
