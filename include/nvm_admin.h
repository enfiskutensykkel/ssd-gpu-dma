#ifndef __NVM_ADMIN_H__
#define __NVM_ADMIN_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <nvm_types.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>



/*
 * Get controller information.
 */
int nvm_admin_ctrl_info(nvm_aq_ref ref,               // AQ pair reference
                        struct nvm_ctrl_info* info,   // Controller information structure
                        void* buffer,                 // Temporary buffer (must be at least 4 KB)
                        uint64_t ioaddr);             // Bus address of buffer as seen by the controller



/* 
 * Get namespace information.
 */
int nvm_admin_ns_info(nvm_aq_ref ref,                 // AQ pair reference
                      struct nvm_ns_info* info,       // NVM namespace information
                      uint32_t ns_id,                 // Namespace identifier
                      void* buffer,                   // Temporary buffer (must be at least 4 KB)
                      uint64_t ioaddr);               // Bus address of buffer as seen by controller



/*
 * Make controller allocate and reserve queues.
 */
int nvm_admin_set_num_queues(nvm_aq_ref ref, uint16_t n_cqs, uint16_t n_sqs);


/*
 * Retrieve the number of allocated queues.
 */
int nvm_admin_get_num_queues(nvm_aq_ref ref, uint16_t* n_cqs, uint16_t* n_sqs);


/*
 * Make controller allocate number of queues before issuing them.
 */
int nvm_admin_request_num_queues(nvm_aq_ref ref, uint16_t* n_cqs, uint16_t* n_sqs);


/*
 * Create IO completion queue (CQ)
 * Caller must set queue memory to zero manually.
 */
int nvm_admin_cq_create(nvm_aq_ref ref,               // AQ pair reference
                        nvm_queue_t* cq,              // CQ descriptor
                        uint16_t id,                  // Queue identifier
                        void* qmem,                   // Queue memory (virtual memory)
                        uint64_t ioaddr);             // Bus address to queue memory as seen by controller


/*
 * Delete IO completion queue (CQ)
 * After calling this, the queue is no longer used and must be recreated.
 * All associated submission queues must be deleted first.
 */
int nvm_admin_cq_delete(nvm_aq_ref ref, nvm_queue_t* cq);



/*
 * Create IO submission queue (SQ)
 * Caller must set queue memory to zero manually.
 */
int nvm_admin_sq_create(nvm_aq_ref ref,               // AQ pair reference
                        nvm_queue_t* sq,              // SQ descriptor
                        const nvm_queue_t* cq,        // Descriptor to paired CQ
                        uint16_t id,                  // Queue identifier
                        void* qmem,                   // Queue memory (virtual)
                        uint64_t ioaddr);             // Bus address to queue as seen by controller



/*
 * Delete IO submission queue (SQ)
 * After calling this, the queue is no longer used and must be recreated.
 */
int nvm_admin_sq_delete(nvm_aq_ref ref, 
                        nvm_queue_t* sq, 
                        const nvm_queue_t* cq);


#ifdef __cplusplus
}
#endif
#endif /* #ifdef __NVM_ADMIN_H__ */
