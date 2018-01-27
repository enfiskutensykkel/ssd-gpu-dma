#ifndef __DIS_NVM_EXAMPLES_SEGMENT_H__
#define __DIS_NVM_EXAMPLES_SEGMENT_H__
#ifdef __DIS_CLUSTER__

#include <nvm_types.h>
#include <stddef.h>
#include <stdint.h>
#include <sisci_types.h>


/*
 * Wrapper for SISCI segment descriptors.
 */
struct segment
{
    uint32_t            id;         // Segment ID
    sci_desc_t          sd;         // SISCI virtual device
    sci_local_segment_t segment;    // Local segment descriptor
    size_t              size;       // Size of segment
};



/*
 * Create a local segment and initialize the wrapper struct.
 */
int segment_create(struct segment* segment, uint32_t segment_id, size_t size);


/*
 * Remove the local segment.
 */
void segment_remove(struct segment* segment);



/*
 * Create a DMA window for the segment.
 */
int dma_create(nvm_dma_t** dma_window, const nvm_ctrl_t* ctrl, struct segment* segment, uint32_t dis_adapter);



/*
 * Destroy the DMA window.
 */
void dma_remove(nvm_dma_t* dma_window, struct segment* segment, uint32_t dis_adapter);


#endif // __DIS_CLUSTER__
#endif // __DIS_NVM_EXAMPLES_SEGMENT_H__
