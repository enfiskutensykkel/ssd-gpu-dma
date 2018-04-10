#ifndef __LIBNVM_SAMPLES_IDENTIFY_H__
#define __LIBNVM_SAMPLES_IDENTIFY_H__

#include <nvm_types.h>
#include <stdint.h>


/*
 * Reset controller and take ownership of admin queues.
 * DMA window must be at least 2 pages.
 * Caller must manually destroy the admin reference.
 */
nvm_aq_ref reset_ctrl(const nvm_ctrl_t* ctrl, const nvm_dma_t* dma_window);


/*
 * Identify controller and print information.
 */
int identify_ctrl(nvm_aq_ref admin, void* ptr, uint64_t ioaddr);


/*
 * Identify namespace and print information.
 */
int identify_ns(nvm_aq_ref admin, uint32_t nvm_namespace, void* ptr, uint64_t ioaddr);


#endif
