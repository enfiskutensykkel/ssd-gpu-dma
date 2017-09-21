#ifndef __DIS_NVM_CTRL_H__
#define __DIS_NVM_CTRL_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <nvm_types.h>
#include <stddef.h>
#include <stdint.h>

#define NVM_CTRL_MEM_MINSIZE   0x2000


/* 
 * Initialize controller reference.
 *
 * Read from PCI device registers and initialize controller reference.
 *
 * Note: ctrl_mem must be at least NVM_CTRL_MEM_MINSIZE large and mapped
 *       as IO memory. See arguments for mmap() for more info.
 *
 * Note: This function should only be called directly if SISCI is not available,
 *       otherwise it is called implicitly.
 */
int nvm_ctrl_init_userspace(nvm_ctrl_t* ctrl, volatile void* ctrl_mem, size_t ctrl_mem_size);


/*
 * Initialize controller reference.
 *
 * Read from device registers and initialize controller reference.
 */
int nvm_ctrl_init(nvm_ctrl_t* ctrl, uint64_t dev_id);


#ifdef __DIS_CLUSTER__

/* 
 * Initialize controller reference.
 *
 * Read from device registers and initialize controller reference.
 */
int nvm_dis_ctrl_init(nvm_ctrl_t* ctrl, uint64_t smartio_dev_id, uint32_t dis_adapter);

#endif


/*
 * Release controller reference.
 */
void nvm_ctrl_free(nvm_ctrl_t ctrl);


/* 
 * Reset NVM controller.
 *
 * The queue memory must be memset to zero and be exactly one page size large.
 * IO addresses must align to the controller page size. 
 *
 * Note: The controller must be unbound from any driver first.
 *
 * Note: This function is implicitly called by the controller manager.
 */
int nvm_ctrl_reset(nvm_ctrl_t ctrl, uint64_t acq_ioaddr, uint64_t asq_ioaddr);


#ifdef __cplusplus
}
#endif
#endif /* __DIS_NVM_CTRL_H__ */
