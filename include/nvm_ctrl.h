#ifndef __DIS_NVM_CTRL_H__
#define __DIS_NVM_CTRL_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifdef _SISCI
#include <sisci_types.h>
#endif

struct nvm_controller;
struct nvm_queue;


#define NVM_CTRL_MEM_MINSIZE   0x2000


/* 
 * Read from PCI device registers and initialize controller reference.
 *
 * Note: ctrl_mem must be at least NVM_CTRL_MEM_MINSIZE large and mapped
 *       as IO memory.
 *
 * Note: This function should only be called if SISCI is not available.
 */
int nvm_ctrl_init_raw(struct nvm_controller* ctrl, volatile void* ctrl_mem, size_t ctrl_mem_size);


/*
 * Remove controller reference.
 */
void nvm_ctrl_free(struct nvm_controller* ctrl);


/* 
 * Reset NVM controller.
 *
 * The queues must be memset to zero and exactly one page size large.
 * The addresses must align to controller pages.
 *
 * Note: This function should not be called directly, use an NVM manager instead.
 */
int nvm_ctrl_reset(const struct nvm_controller* ctrl, uint64_t acq_addr, uint64_t asq_addr);


/*
 * Initialize NVM queue handle.
 *
 * Note: The caller must memset the queue to zero.
 * Note: size of queue must be exactly one page size large
 */
void nvm_queue_clear(struct nvm_queue* q, const struct nvm_controller* ctrl, int cq, uint16_t no, void* vaddr, uint64_t ioaddr);


#ifdef _SISCI

/* 
 * Read from PCI device registers and initialize controller reference.
 */
int nvm_ctrl_init(struct nvm_controller* ctrl, uint64_t smartio_dev_id, uint32_t dis_adapter);


/*
 * Map local segment for an NVM controller.
 */
int nvm_dma_window_create(sci_local_segment_t segment, const struct nvm_controller* ctrl, uint64_t* ioaddr);


/*
 * Unmap local segment for an NVM controller.
 */
int nvm_dma_window_destroy(sci_local_segment_t segment, const struct nvm_controller* ctrl);

#endif


#ifdef __cplusplus
}
#endif
#endif
