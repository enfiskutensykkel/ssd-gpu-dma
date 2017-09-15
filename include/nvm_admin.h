#ifndef __DIS_NVM_ADMIN_H__
#define __DIS_NVM_ADMIN_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

struct nvm_queue;
struct nvm_command;


/* List of NVM admin command opcodes */
enum nvm_admin_command_set
{
    NVM_ADMIN_DELETE_SUBMISSION_QUEUE   = (0x00 << 7) | (0x00 << 2) | 0x00,
    NVM_ADMIN_CREATE_SUBMISSION_QUEUE   = (0x00 << 7) | (0x00 << 2) | 0x01,
    NVM_ADMIN_DELETE_COMPLETION_QUEUE   = (0x00 << 7) | (0x01 << 2) | 0x00,
    NVM_ADMIN_CREATE_COMPLETION_QUEUE   = (0x00 << 7) | (0x01 << 2) | 0x01,
    NVM_ADMIN_IDENTIFY                  = (0x00 << 7) | (0x01 << 2) | 0x02,
    NVM_ADMIN_ABORT                     = (0x00 << 7) | (0x02 << 2) | 0x00,
    NVM_ADMIN_SET_FEATURES              = (0x00 << 7) | (0x02 << 2) | 0x01,
    NVM_ADMIN_GET_FEATURES              = (0x00 << 7) | (0x02 << 2) | 0x02
};



/*
 * Create IO completion queue (CQ).
 *
 * Build an NVM admin command for creating a CQ.
 */
void nvm_admin_cq_create(struct nvm_command* cmd, const struct nvm_queue* cq);


/* 
 * Create IO submission queue (SQ).
 *
 * Build an NVM admin command for creating an SQ. Note that the associated
 * CQ must have been created first.
 */
void nvm_admin_sq_create(struct nvm_command* cmd, const struct nvm_queue* sq, const struct nvm_queue* cq);


/*
 * Delete IO submission queue (SQ).
 *
 * Build an NVM admin command for deleting an SQ.
 */
void nvm_admin_sq_delete(struct nvm_command* cmd, const struct nvm_queue* sq, const struct nvm_queue* cq);


/*
 * Delete IO completion queue (CQ).
 *
 * Build an NVM admin command for deleting a CQ. Note that the associated
 * SQ must have been deleted first.
 */
void nvm_admin_cq_delete(struct nvm_command* cmd, const struct nvm_queue*cq);


/* 
 * Identify controller.
 *
 * Build an NVM admin command for identifying the controller.
 */
void nvm_admin_identify_ctrl(struct nvm_command* cmd, uint64_t ioaddr);


/*
 * Identify namespace.
 */
void nvm_admin_identify_ns(struct nvm_command* cmd, uint32_t ns_id, uint64_t ioaddr);


/*
 * Set/get current number of queues.
 */
void nvm_admin_current_num_queues(struct nvm_command* cmd, int set, uint16_t n_cqs, uint16_t n_sqs);


#ifdef __cplusplus
}
#endif
#endif /* #ifdef __DIS_NVM_ADMIN_H__ */
