#ifndef __NVM_INTERNAL_RPC_H__
#define __NVM_INTERNAL_RPC_H__

#include <nvm_types.h>
#include <nvm_manager.h>
#include <stddef.h>
#include <stdint.h>


#ifdef _SISCI
#include <sisci_types.h>
#define DIS_CLUSTER_TIMEOUT     2500
#endif


/*
 * RPC command message format.
 */
struct __attribute__((packed)) rpc_cmd
{
    uint32_t                    node_id;    // Initiator DIS node ID
    uint32_t                    intr_no;    // Initiator "callback" interrupt
    unsigned char               cmd[64];    // Command to execute
};



/*
 * RPC completion message format.
 */
struct __attribute__((packed)) rpc_cpl
{
    int                         status;     // Status from the local routine
    int                         accepted;   // Indicates whether or not command was accepted
    unsigned char               cpl[16];    // Command completion
};



#ifdef _SISCI

/* 
 * RPC handle descriptor.
 *
 * Descriptor to local interrupt other nodes can connect to.
 */
struct rpc_handle
{
    sci_desc_t                  sd;         // SISCI virtual device
    sci_local_data_interrupt_t  intr;       // Local data interrupt
    uint32_t                    intr_no;    // Interrupt number
    uint32_t                    adapter;    // Local DIS adapter
    nvm_rpc_filter_t            filter;     // Filter callback
};



/*
 * RPC reference.
 *
 * Reference to a remote interrupt the node is connected to.
 */
struct rpc_reference
{
    sci_desc_t                  sd;         // SISCI virtual device
    sci_remote_data_interrupt_t intr;       // Remote data interrupt
    uint32_t                    intr_no;    // Interrupt number
    uint32_t                    node_id;    // Remote DIS node id
    uint32_t                    adapter;    // Local DIS adapter number
};



/*
 * List of RPC handle descriptor.
 *
 * Linked list of RPC handles.
 */
struct rpc_list
{
    struct rpc_list*    next;               // Pointer to the next in list
    struct rpc_handle   handle;             // Actual handle
};



/*
 * Initialize a RPC handle descriptor.
 * Open descriptor and create local data interrupt.
 */
int _nvm_rpc_handle_init(struct rpc_handle* handle, uint32_t adapter);



/*
 * Destroy an RPC handle descriptor.
 * Destroy local data interrupt and close descriptor.
 */
void _nvm_rpc_handle_free(struct rpc_handle* handle);



/*
 * Initialize a remote RPC reference.
 * Open descriptor and connect to remote data interrupt.
 */
int _nvm_rpc_ref_init(struct rpc_reference* ref, uint32_t node_id, uint32_t intr_no, uint32_t adapter);



/*
 * Destroy RPC reference.
 * Disconnect from remote data interrupt and close descriptor.
 */
void _nvm_rpc_ref_free(struct rpc_reference* ref);



/*
 * Trigger remote interrupt with data.
 */
int _nvm_rpc_ref_send(struct rpc_reference* ref, void* data, size_t length);


#endif



/*
 * Execute an NVM admin command using a local manager.
 */
int _nvm_rpc_local(nvm_manager_t mngr, const struct rpc_cmd* cmd, struct rpc_cpl* cpl);


#endif /* __NVM_INTERNAL_RPC_H__ */
