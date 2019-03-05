#ifndef __NVM_INTERNAL_DMA_H__
#define __NVM_INTERNAL_DMA_H__

#include <nvm_types.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>


/* Forward declaration */
struct va_range;



/*
 * Callback type for freeing an address range descriptor.
 * Called after the range is unmapped for the device and virtual address mapping can
 * be released.
 */
typedef void (*va_range_free_t)(struct va_range* va);



/*
 * Virtual address range descriptor.
 * This structure describes a custom address range mapped in userspace.
 */
struct va_range
{
    volatile void*  vaddr;      // Virtual address of mapped address range
    size_t          page_size;  // Alignment of mapping (page size)
    size_t          n_pages;    // Number of pages for address range
};


#define VA_RANGE_INIT(vaddr, page_size, n_pages)    \
    (struct va_range) {(vaddr), (page_size), (n_pages)}


/*
 * Map address range for a controller and create and initialize a DMA handle.
 */
int _nvm_dma_init(nvm_dma_t** handle,
                  const nvm_ctrl_t* ctrl,
                  struct va_range* va,
                  bool remote,
                  va_range_free_t release);


#endif /* __NVM_INTERNAL_DMA_H__ */
