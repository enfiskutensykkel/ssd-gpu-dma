#ifndef __NVM_INTERNAL_DMA_H__
#define __NVM_INTERNAL_DMA_H__

#include <nvm_types.h>
#include <stddef.h>
#include <stdint.h>

/* Forward declaration */
struct dma_map;


/* Callback type for unmapping and deleting custom mapping */
typedef void (*dma_map_free_t)(struct dma_map* map);



/*
 * DMA mapping descriptor.
 * 
 * This structure describes a custom address range mapping.
 * Custom types should include this structure in head.
 */
struct __attribute__((aligned (64))) dma_map
{
    void*           vaddr;      // Virtual address of mapped address range
    size_t          page_size;  // Page size of address range
    size_t          n_pages;    // Number of pages for address range
};



/*
 * Create a DMA handle container.
 */
int _nvm_dma_create(nvm_dma_t** handle,
                    const nvm_ctrl_t* ctrl,
                    struct dma_map* map,
                    dma_map_free_t release);


/*
 * Initialize DMA handle.
 */
void _nvm_dma_handle_populate(nvm_dma_t* handle, 
                              const nvm_ctrl_t* ctrl, 
                              const uint64_t* ioaddrs);



/*
 * Invoke release callback and remove a DMA handle container.
 */
void _nvm_dma_remove(nvm_dma_t* handle);



#endif /* __NVM_INTERNAL_DMA_H__ */
