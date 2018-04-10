#ifndef __NVM_INTERNAL_RPC_H__
#define __NVM_INTERNAL_RPC_H__

#include <nvm_types.h>
#include <nvm_queue.h>
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>


/* Forward declaration */
struct nvm_admin_reference;



/*
 * Callback to delete custom instance data.
 */
typedef void (*rpc_deleter_t)(void* data, uint32_t key, int remaining_handles);



/*
 * RPC client-side stub definition.
 * 
 * Should perform the following actions.
 *      - marshal command
 *      - send command to remote host
 *      - wait for completion (or timeout)
 *      - unmarshal completion and return status
 */
typedef int (*rpc_stub_t)(void*, nvm_cmd_t*, nvm_cpl_t*);



/*
 * Allocate a reference wrapper and increase controller reference.
 */
int _nvm_ref_get(nvm_aq_ref* handle, const nvm_ctrl_t* ctrl);



/*
 * Free reference wrapper and decrease controller reference.
 */
void _nvm_ref_put(nvm_aq_ref ref);



/*
 * Insert binding handle to server's list of handles.
 * If key is already found, this function will fail.
 */
int _nvm_rpc_handle_insert(nvm_aq_ref ref, uint32_t key, void* data, rpc_deleter_t release);



/*
 * Remove local binding handle.
 * This function will call the release callback.
 */
void _nvm_rpc_handle_remove(nvm_aq_ref ref, uint32_t key);



/*
 * Bind reference to remote handle.
 */
int _nvm_rpc_bind(nvm_aq_ref ref, void* data, rpc_deleter_t deleter, rpc_stub_t stub);



/*
 * Execute a local admin command.
 */
int _nvm_local_admin(nvm_aq_ref ref, const nvm_cmd_t* cmd, nvm_cpl_t* cpl);



#endif /* __NVM_INTERNAL_RPC_H__ */
