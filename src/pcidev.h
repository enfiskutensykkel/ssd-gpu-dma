#ifndef __DIS_NVM_INTERNAL_PCIDEV_H__
#define __DIS_NVM_INTERNAL_PCIDEV_H__
#ifdef _SISCI /* Only expose the following if SISCI is used */

#include <stddef.h>
#include <stdint.h>
#include <sisci_types.h>


/* PCI device BAR memory
 *
 * Represents a handle to a memory-mapped BAR region of a PCI device.
 */
struct pci_dev
{
    uint64_t                device_id;      // SISCI SmartIO unique device ID
    uint32_t                adapter;        // DIS adapter number
    sci_desc_t              sd;             // SISCI virtual device
    sci_remote_segment_t    segment;        // SISCI remote segment to device BAR
    sci_map_t               map;            // SISCI memory map handle
    sci_device_t            device;         // SISCI SmartIO device handle
    size_t                  mm_size;        // Size of memory-mapped region
    volatile void*          mm_ptr;         // Memory-mapped pointer to device BAR
};


/*
 * Acquire a PCI device BAR reference.
 */
int pci_dev_ref_get(struct pci_dev* ref, uint64_t smartio_device_id, uint32_t dis_adapter, int bar, size_t size);


/* 
 * Release a PCI device BAR reference.
 */
void pci_dev_ref_put(struct pci_dev* ref);


#endif /* #ifdef _SISCI */
#endif /* __DIS_NVM_ADMIN_H__ */
