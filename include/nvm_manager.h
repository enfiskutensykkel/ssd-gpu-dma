#ifndef __DIS_NVM_MANAGER_H__
#define __DIS_NVM_MANAGER_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <nvm_types.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>


#ifdef __DIS_CLUSTER__

/*
 * Remote command accepter callback.
 *
 * Callback function called whenever a remote NVM admin command is received.
 * Should indicate whether or not a remote admin command should be accepted or
 * dismissed by returning true for accept or false for dismiss. Note that the
 * callback is also allowed to make changes to the command, although no 
 * no signalling mechanism is in place for notifying the remote caller of any
 * changes.
 */
typedef bool (*nvm_rpc_filter_t)(uint32_t dis_node_id, uint32_t dis_adapter, uint32_t dis_intr_no, nvm_cmd_t* cmd);

#endif /* _SISCI */


/* 
 * Register as controller manager.
 *
 * Register the local process as the dedicated controller manager, responsible 
 * for exclusive access to the NVM admin queues (ASQ and ACQ). This function
 * will initialize queue descriptors and reset the controller. Make sure that
 * the DMA window for the queue memory is at least two pages large.
 *
 * Access to the admin queues are available through an RPC-like interface.
 *
 * Note: This function will implicitly reset the controller.
 */
int nvm_manager_register(nvm_manager_t* manager, nvm_ctrl_t ctrl, nvm_dma_t queue_memory);


/*
 * Unregister as controller manager.
 *
 * Stop relaying admin commands and release resources.
 */
void nvm_manager_unregister(nvm_manager_t manager);



#ifdef __DIS_CLUSTER__

/* 
 * Enable remote commands.
 *
 * Enables relaying of remote admin commands on the specified DIS adapter.
 * The accepter callback indicates on an per command basis whether or not the
 * command should be relayed to the controller or not.
 *
 * The manager will relay NVM admin commands from remote hosts to the local 
 * admin SQ, and send the corresponding completion back to the remote 
 * command initiator. There should be only one manager per controller in a
 * cluster. However, the manager must be manually enabled on the desired
 * DIS adapters.
 *
 */
int nvm_dis_rpc_enable(nvm_manager_t manager, uint32_t dis_adapter, uint32_t dis_intr_no, nvm_rpc_filter_t filter);


/*
 * Disable remote commands.
 *
 * Stop relaying admin commands from remote nodes connected to the specified DIS adapter.
 */
int nvm_dis_rpc_disable(nvm_manager_t manager, uint32_t dis_adapter);


#endif 



#ifdef __cplusplus
}
#endif
#endif /* #ifdef __DIS_NVM_MANAGER_H__ */
