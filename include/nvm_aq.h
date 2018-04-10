#ifndef __NVM_AQ_H__
#define __NVM_AQ_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <nvm_types.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>



/*
 * Create admin queue pair
 *
 * Take exclusive ownership of an NVM controller. This function resets the 
 * controller and configures NVM admin queues. 
 *
 * Returns a reference handle that can be used for admin RPC calls.
 */
int nvm_aq_create(nvm_aq_ref* ref, const nvm_ctrl_t* ctrl, const nvm_dma_t* aq_mem);



/*
 * Destroy admin queues and references.
 *
 * Send NVM abort command to controller and deallocate admin queues.
 *
 * After calling this function, all admin queue references are invalid.
 * This also means that remote references will no longer be valid.
 *
 * This function will also work for unbinding remote references.
 */
void nvm_aq_destroy(nvm_aq_ref ref);



//int nvm_tcp_rpc_enable(nvm_aq_ref ref, uint16_t port, nvm_rpc_cb_t filter, void* data);
//int nvm_tcp_rpc_disable(nvm_aq_ref ref, uint16_t port);



#ifdef __DIS_CLUSTER__

/*
 * Maximum number of adapters.
 */
#define NVM_DIS_RPC_MAX_ADAPTER     4


/*
 * Callback function invoked whenever a remote NVM admin command is received.
 * Should indicate whether or not a remote admin command is accepted and can
 * be enqueued by using the return value.
 *
 * The remote command can also be modified if necessary.
 */
typedef bool (*nvm_dis_rpc_cb_t)(nvm_cmd_t* cmd, uint32_t dis_adapter, uint32_t dis_node_id);



/*
 * Enable remote admin commands.
 * Allows remote processes to relay NVM admin commands to the local process.
 */
int nvm_dis_rpc_enable(nvm_aq_ref ref,               // NVM admin queue-pair reference
                       uint32_t dis_adapter,         // Local adapter to enable interrupt on
                       nvm_dis_rpc_cb_t filter);     // Filter callback (can be NULL)



/*
 * Disable remote admin commands.
 * Stop processing admin commands from remote processes.
 */
void nvm_dis_rpc_disable(nvm_aq_ref ref, uint32_t dis_adapter);

#endif /* __DIS_CLUSTER__ */



#ifdef __cplusplus
}
#endif
#endif /* #ifdef __NVM_AQ_H__ */
