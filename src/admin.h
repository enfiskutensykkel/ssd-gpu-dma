#ifndef __NVM_INTERNAL_ADMIN_H__
#define __NVM_INTERNAL_ADMIN_H__



/*
 * Create IO completion queue (CQ).
 *
 * Build an NVM admin command for creating a CQ.
 */
void _nvm_admin_cq_create(nvm_cmd_t* cmd, const nvm_queue_t* cq);



/* 
 * Create IO submission queue (SQ).
 *
 * Build an NVM admin command for creating an SQ. Note that the associated
 * CQ must have been created first.
 */
void _nvm_admin_sq_create(nvm_cmd_t* cmd, const nvm_queue_t* cq,  const nvm_queue_t* sq);



/*
 * Delete IO submission queue (SQ).
 *
 * Build an NVM admin command for deleting an SQ.
 */
void _nvm_admin_sq_delete(nvm_cmd_t* cmd, const nvm_queue_t* sq);



/*
 * Delete IO completion queue (CQ).
 *
 * Build an NVM admin command for deleting a CQ. Note that the associated
 * SQ must have been deleted first.
 */
void _nvm_admin_cq_delete(nvm_cmd_t* cmd, const nvm_queue_t* cq);



/* 
 * Identify controller.
 *
 * Build an NVM admin command for identifying the controller.
 */
void _nvm_admin_identify_ctrl(nvm_cmd_t* cmd, uint64_t ioaddr);



/*
 * Identify namespace.
 */
void _nvm_admin_identify_ns(nvm_cmd_t* cmd, uint32_t ns_id, uint64_t ioaddr);



/*
 * Set/get current number of queues.
 */
void _nvm_admin_current_num_queues(nvm_cmd_t* cmd, bool set, uint16_t n_cqs, uint16_t n_sqs);



#endif /* __NVM_INTERNAL_ADMIN_H__ */
