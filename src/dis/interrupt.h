#ifndef __NVM_INTERNAL_DIS_INTERRUPT_H__
#define __NVM_INTERNAL_DIS_INTERRUPT_H__
#ifdef _SISCI

/* Make sure everything is defined as needed */
#ifndef __DIS_CLUSTER__
#define __DIS_CLUSTER__
#endif

/* Necessary includes */
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <sisci_types.h>



/* Forward declarations */
struct local_intr;
struct remote_intr;



/*
 * Interrupt callback.
 */
typedef void (*intr_callback_t)(void* user_data, void* recv_data, uint16_t length);



/*
 * Local interrupt descriptor.
 * Data must be free'd manually.
 */
struct local_intr
{
    sci_desc_t                  sd;         // SISCI virtual device descriptor
    sci_local_data_interrupt_t  intr;       // SISCI data interrupt handle
    uint32_t                    adapter;    // DIS adapter
    uint32_t                    intr_no;    // Interrupt number
    uint32_t                    node_id;    // DIS node identifier
    void*                       data;       // User data
    intr_callback_t             callback;   // Interrupt callback
};



/*
 * Remote interrupt descriptor.
 */
struct remote_intr
{
    sci_desc_t                  sd;         // SISCI virtual device descriptor
    sci_remote_data_interrupt_t intr;       // SISCI data interrupt reference
};



/*
 * Create a local data interrupt.
 */
int _nvm_local_intr_get(struct local_intr* intr,
                        uint32_t adapter, 
                        void* cb_data, 
                        intr_callback_t cb_func);



/*
 * Remove a local data interrupt.
 */
void _nvm_local_intr_put(struct local_intr* intr);



/*
 * Block for a duration while waiting for an interrupt and removes interrupt afterwards.
 * Returns success if length of received data matches expected length.
 */
int _nvm_local_intr_wait(struct local_intr* intr, void* data, uint16_t maxlen, uint32_t timeout);



/*
 * Connect to remote interrupt.
 */
int _nvm_remote_intr_get(struct remote_intr* intr, uint32_t adapter, uint32_t node_id, uint32_t intr_no);



/*
 * Disconnect from remote interrupt.
 */
void _nvm_remote_intr_put(struct remote_intr* intr);



/*
 * Trigger remote interrupt with data.
 */
int _nvm_remote_intr_trigger(const struct remote_intr* intr, void* data, uint16_t len);



/*
 * Connect to remote interrupt, send data, and disconnect.
 */
int _nvm_remote_intr_fire_and_forget(uint32_t adapter, 
                                     uint32_t node_id, 
                                     uint32_t intr_no, 
                                     void* data, 
                                     uint16_t len);

#endif /* _SISCI */
#endif /* __NVM_INTERNAL_DIS_INTERRUPT_H__ */
