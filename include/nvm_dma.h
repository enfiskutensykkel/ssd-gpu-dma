#ifndef __DIS_NVM_DMA_H__
#define __DIS_NVM_DMA_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <nvm_types.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __DIS_CLUSTER__
#include <sisci_types.h>
#endif


/*
 * Create DMA window descriptor.
 *
 * Create a DMA window descriptor, describing a region of memory that is 
 * accessible from the NVM controller. As the host's page size does not 
 * necessarily map to the controller's page size (MPS), this function will 
 * calculate the necessary offsets into the actual memory pages.
 *
 * The caller must also supply the physical/bus addresses of the physical
 * memory pages, as well as the page size and total number of pages. While
 * virtual memory is assumed to be continuous, physical pages do not need to be 
 * contiguous (Note: it is possible to pass NULL for the vaddr argument).
 *
 * Addresses must be aligned to controller page size (MPS).
 */
// TODO: rename to nvm_dma_map
int nvm_dma_window_init(nvm_dma_t* wnd, nvm_ctrl_t ctrl, void* vaddr, size_t page_size, size_t n_pages, uint64_t* paddrs);


// TODO: rename to nvm_dma_map_host
int nvm_dma_window_host_map(nvm_dma_t* wnd, nvm_ctrl_t ctrl, void* vaddr, size_t size);



#if ( defined( __CUDA__ ) || defined( __CUDACC__ ) )

int nvm_dma_window_device_map(nvm_dma_t* wnd, nvm_ctrl_t ctrl, void* devptr, size_t size);

#endif


#if defined( __DIS_CLUSTER__ )

/*
 * Create DMA window descriptor from SISCI segment.
 *
 * Create a DMA window descriptor from a local segment, and reverse-map it
 * making it accessible for the controller. As segment memory is always
 * continuous, it is not necessary for the caller to calculate physical memory
 * addresses. The size, however, should be controller page size aligned.
 *
 * Note that the controller must "be alive" as long as DMA windows exists,
 * because the reference is used to unmap windows and release resources.
 */
// TODO: rename to nvm_dis_dma_map
int nvm_dis_dma_window_init(nvm_dma_t* wnd, nvm_ctrl_t ctrl, uint32_t dis_adapter, sci_local_segment_t segment, void* vaddr, size_t size);

#endif


/*
 * Remove DMA window descriptor.
 *
 * Remove DMA descriptor, and unmap it for the controller if necessary.
 * TODO: rename to nvm_dma_unmap
 */
void nvm_dma_window_free(nvm_dma_t wnd);


#ifdef __cplusplus
}
#endif
#endif /* __DIS_NVM_DMA_H__ */
