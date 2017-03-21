#ifndef __PCI_CONFIG__
#define __PCI_CONFIG__
#ifdef __cplusplus
extern "C" {
#endif


/* Set the appropriate registers in device's config space
 *
 * Set the BusMaster bit in the COMMAND register in order for the device to
 * be able to do DMA.
 */
int pci_set_config(int domain, int bus, int device, int function);


/* Open file descriptor to a device's BAR regions */
int pci_open_bar(int domain, int bus, int device, int function, int bar);


#ifdef __cplusplus
}
#endif
#endif
