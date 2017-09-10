#ifndef __DIS_NVM_RPC_H__
#define __DIS_NVM_RPC_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <nvm_types.h>
#include <stdint.h>


/*
 * Bind the manager reference to a local manager.
 */
int nvm_rpc_bind_local(nvm_rpc_t* ref, nvm_manager_t manager);


#ifdef __DIS_CLUSTER__

/*
 * Bind the manager reference to a remote manager.
 */
int nvm_dis_rpc_bind(nvm_rpc_t* ref, uint32_t remote_node_id, uint32_t remote_intr_no, uint32_t local_adapter);

#endif


/*
 * Unbind the manager reference.
 */
void nvm_rpc_unbind(nvm_rpc_t ref);


/*
 * Relay NVM admin command to the controller manager.
 *
 * Call this function to relay an NVM command to the controller manager.
 * The manager is responsible for submitting the command to the physical 
 * controller and sending the completion back. This function will block until
 * either a timeout occurs, or until a completion is received, and the caller
 * should use a timeout value much greater than the controller timeout.
 *
 * Note: A remote manager is allowed to change the command, and can do so 
 * without notifying the command initiator as no signalling mechanism is in
 * place.
 */
int nvm_rpc_raw_cmd(nvm_rpc_t ref, const nvm_cmd_t* cmd, nvm_cpl_t* cpl, uint64_t timeout);


/*
 * Identify controller.
 */
int nvm_rpc_identify(nvm_rpc_t ref, nvm_ctrl_t ctrl, nvm_dma_t dma_window, nvm_ctrl_info_t* info);


/*
 * Create IO completion queue (CQ)
 */
int nvm_rpc_cq_create(nvm_rpc_t ref, nvm_ctrl_t ctrl, nvm_dma_t dma_window, nvm_queue_t* cq);


/*
 * Create IO submission queue (SQ)
 */
int nvm_rpc_sq_create(nvm_rpc_t ref, nvm_ctrl_t ctrl, const nvm_queue_t* cq, nvm_dma_t dma_window, nvm_queue_t* sq);


#ifdef __cplusplus
}
#endif
#endif /* #ifdef __DIS_NVM_RPC_H__ */
