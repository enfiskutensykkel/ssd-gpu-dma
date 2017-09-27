#ifndef __DIS_NVM_RPC_H__
#define __DIS_NVM_RPC_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <nvm_types.h>
#include <stdint.h>


/*
 * Bind the manager reference to a local manager.
 * TODO: rename to nvm_rpc_bind
 */
int nvm_rpc_bind_local(nvm_rpc_t* ref, nvm_manager_t manager);


#ifdef __DIS_CLUSTER__

/*
 * Bind the manager reference to a remote manager.
 */
int nvm_dis_rpc_bind(nvm_rpc_t* ref, uint32_t remote_node_id, uint32_t remote_intr_no, uint32_t local_adapter);

#endif


#ifdef __TCP_CLUSTER__
// int nvm_tcp_rpc_bind(nvm_rpc_t* ref, const char* hostname, uint16_t port);
#endif


/*
 * Unbind the manager reference.
 */
void nvm_rpc_unbind(nvm_rpc_t ref);

// TODO: make all rpc commands return int32_t instead
//       pack NVM status code and status type into returnval and 
//       set higher bit to indicate nvm error or errno


/*
 * Relay NVM admin command to the controller manager.
 *
 * Call this function to relay an NVM command to the controller manager.
 * The manager is responsible for submitting the command to the physical 
 * controller and sending the completion back. This function will block until
 * either a timeout occurs, or until a completion is received.
 *
 * Note: A remote manager is allowed to change the command, and can do so 
 * without notifying the command initiator as no signalling mechanism is in
 * place.
 */
int nvm_rpc_raw_cmd(nvm_rpc_t ref, const nvm_cmd_t* cmd, nvm_cpl_t* cpl);


/*
 * Get controller information.
 */
int nvm_rpc_ctrl_info(nvm_ctrl_info_t* info, nvm_rpc_t ref, nvm_ctrl_t ctrl, void* vaddr, uint64_t ioaddr);


/* 
 * Get namespace information.
 */
int nvm_rpc_ns_info(nvm_ns_info_t* info, nvm_rpc_t ref, uint32_t ns_id, void* vaddr, uint64_t ioaddr);


/*
 * Make controller allocate number of queues before issuing them.
 */
int nvm_rpc_set_num_queues(nvm_rpc_t ref, uint16_t n_cqs, uint16_t n_sqs);


/*
 * Retrieve the number of allocated queues.
 */
int nvm_rpc_get_num_queues(nvm_rpc_t ref, uint16_t* n_cqs, uint16_t* n_sqs);


/*
 * Make controller allocate number of queues before issuing them.
 */
int nvm_rpc_request_num_queues(nvm_rpc_t ref, uint16_t* n_cqs, uint16_t* n_sqs);


/*
 * Create IO completion queue (CQ)
 */
int nvm_rpc_cq_create(nvm_queue_t* cq, nvm_rpc_t ref, nvm_ctrl_t ctrl, uint16_t id, void* vaddr, uint64_t ioaddr);


/*
 * Create IO submission queue (SQ)
 */
int nvm_rpc_sq_create(nvm_queue_t* sq, nvm_rpc_t ref, nvm_ctrl_t ctrl, const nvm_queue_t* cq, uint16_t id, void* vaddr, uint64_t ioaddr);



#ifdef __cplusplus
}
#endif
#endif /* #ifdef __DIS_NVM_RPC_H__ */
