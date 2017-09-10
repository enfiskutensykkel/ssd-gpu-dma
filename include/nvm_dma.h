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
int nvm_dma_window_init(nvm_dma_t* wnd, nvm_ctrl_t ctrl, void* vaddr, size_t page_size, size_t n_pages, uint64_t* paddrs);



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
int nvm_dis_dma_window_init(nvm_dma_t* wnd, nvm_ctrl_t ctrl, uint32_t dis_adapter, sci_local_segment_t segment, void* vaddr, size_t size);

//#elif defined( _LKM )
//
//int nvm_dma_window_lkm_ram(nvm_dma_t* wnd, nvm_ctrl_t ctrl, int lkm_ioctl, void* vaddr, size_t size);
//
//int nvm_dma_window_lkm_gpu(nvm_dma_t* wnd, nvm_ctrl_t ctrl, int lkm_ioctl, void* devptr, size_t page_size, size_t n_pages);

#endif


/*
 * Remove DMA window descriptor.
 *
 * Remove DMA descriptor, and unmap it for the controller if necessary.
 */
void nvm_dma_window_free(nvm_dma_t wnd);


#ifdef __cplusplus
}
#endif
#endif /* __DIS_NVM_DMA_H__ */
