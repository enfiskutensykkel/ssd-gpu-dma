#ifndef __DIS_NVM_MANAGER_H__
#define __DIS_NVM_MANAGER_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

struct nvm_manager;
struct nvm_command;
struct nvm_completion;
struct nvm_controller_information;




/* NVM controller manager handle
 *
 * Handle to the controller manager. The controller manager is responsible
 * for administrating the controller's admin queues (ASQ and ACQ) in a 
 * thread-safe manner. Only one manager per controller should be created.
 */
typedef struct nvm_manager* nvm_mngr_t;


/*
 * Initialize a NVM manager
 *
 * Note: At least three pages
 */
int nvm_mngr_init_raw(nvm_mngr_t* mngr, const struct nvm_controller* ctrl, void* vaddr, size_t n_ioaddrs, const uint64_t* ioaddrs);


/*
 * Release and free resources acquired by the controller manager.
 */
void nvm_mngr_free(nvm_mngr_t mngr);


/*
 * Relay an NVM admin command to the controller manager
 *
 * Relay an NVM admin command to the controller manager that controls the admin
 * queues (ASQ and ACQ). The function will block until either a timeout occurs
 * or until the controller manager replies with an NVM completion.
 *
 * Note: Caller should use a timeout of at least twice the controller's timeout
 *
 * Note: This function is very inefficient and should typically only be called
 *       in order to create or delete IO queues
 */
int nvm_local_cmd(nvm_mngr_t mngr, const struct nvm_command* cmd, struct nvm_completion* cpl, uint64_t timeout);


/*
 * Get controller information.
 */
int nvm_ctrl_get_info(nvm_mngr_t mngr, nvm_ctrl_info_t* info);


#ifdef _SISCI /* Only expose the following if SISCI is used */

/*
 * Create NVM controller manager
 *
 * Create admin queues (ASQ and ACQ) and manage NVM admin commands.
 */
int nvm_mngr_init(nvm_mngr_t* mngr, const struct nvm_controller* ctrl, uint32_t unique_id);


/*
 * Export NVM manager and accept remote admin commands on a specified adapter.
 */
int nvm_mngr_export(nvm_mngr_t mngr, uint32_t adapter, uint32_t intno);


/*
 * Stop accepting remote admin commands on a specific adapter.
 */
int nvm_mngr_unexport(nvm_mngr_t mngr, uint32_t adapter);


/*
 * Relay an NVM admin command to the controller manager
 *
 * Relay an NVM admin command to the controller manager that controls the admin
 * queues (ASQ and ACQ). The function will block until either a timeout occurs
 * or until the controller manager replies with an NVM completion.
 *
 * Note: Caller should use a timeout of at least twice the controller's timeout
 *
 * Note: This function is very inefficient and should typically only be called
 *       in order to create or delete IO queues
 */
int nvm_remote_cmd(uint32_t node_id, uint32_t adapter, uint32_t intno, const struct nvm_command* cmd, struct nvm_completion* cpl, uint64_t timeout);


/*
 * Complete a relayed command with a completion
 *
 * Send completion reply back to admin command initiator. 
 *
 * Note: This function is used internally by the controller manager and 
 *       should not be called directly.
 */
int nvm_remote_cpl(uint32_t node_id, uint32_t adapter, uint32_t intno, const struct nvm_completion* cpl, uint64_t timeout);


#endif /* #ifdef _SISCI */

#ifdef __cplusplus
}
#endif
#endif /* #ifdef __DIS_NVM_MNGR_H__ */
