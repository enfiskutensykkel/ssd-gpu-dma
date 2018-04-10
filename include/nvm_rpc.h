#ifndef __NVM_RPC_H__
#define __NVM_RPC_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <nvm_types.h>
#include <stdint.h>


//int nvm_tcp_rpc_bind(nvm_aq_ref* ref, const char* hostname, uint16_t port);



#ifdef __DIS_CLUSTER__

/*
 * Bind admin queue-pair reference to remote handle.
 * The user should call the nvm_aq_destroy() to remove binding.
 */
int nvm_dis_rpc_bind(nvm_aq_ref* ref, const nvm_ctrl_t* ctrl, uint32_t adapter);

#endif



/*
 * Unbind admin queue-pair reference.
 * If reference is not bound (i.e., it is local), this function will do nothing.
 */
void nvm_rpc_unbind(nvm_aq_ref ref);



/*
 * Relay NVM admin command.
 *
 * Use a local AQ pair reference to relay a NVM admin command to ASQ and get
 * a corresponding completion from the ACQ. This function will block until 
 * either a timeout occurs or until the command is completed.
 *
 * Return value:
 * - If return value is zero, it indicates success.
 * - If return value is positive, it indicates an errno.
 * - If return value is negative, it indicates an NVM error.
 *
 * Use the error handling macros in nvm_error.h
 *
 * Note: The command can be modified.
 */
int nvm_raw_rpc(nvm_aq_ref ref, nvm_cmd_t* cmd, nvm_cpl_t* cpl);



#ifdef __cplusplus
}
#endif
#endif /* #ifdef __NVM_RPC_H__ */
