#ifdef _SISCI
#include <sisci_api.h>
#include <sisci_types.h>
#include <sisci_error.h>
#ifndef __DIS_CLUSTER__
#define __DIS_CLUSTER__
#endif
#endif

#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_aq.h>
#include <nvm_rpc.h>
#include <nvm_queue.h>
#include <nvm_ctrl.h>
#include <nvm_cmd.h>
#include <nvm_util.h>
#include <nvm_error.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
#include "dis/device.h"
#include "rpc.h"
#include "ctrl.h"
#include "util.h"
#include "dprintf.h"



/*
 * Local admin queue-pair descriptor.
 */
struct local_admin
{
    nvm_queue_t         acq;        // Admin completion queue (ACQ)
    nvm_queue_t         asq;        // Admin submission queue (ASQ)
    uint64_t            timeout;    // Controller timeout
};



/*
 * Linked list of RPC server-side binding handles.
 */
struct rpc_handle
{
    struct rpc_handle*  next;       // Pointer to next handle in list
    uint32_t            key;        // Handle key
    void*               data;       // Custom instance data
    rpc_deleter_t       release;    // Callback to release the instance data
};



/*
 * Administration queue-pair reference.
 *
 * Represents either a reference to a remote descriptor, or is a local 
 * descriptor. In other words, this handle represents both RPC clients and
 * RPC servers.
 */
struct nvm_admin_reference
{
    const nvm_ctrl_t*       ctrl;       // Controller reference
    pthread_mutex_t         lock;       // Ensure exclusive access to the reference
    int                     n_handles;  // Number of handles
    struct rpc_handle*      handles;    // Linked list of binding handles (if server)
    void*                   data;       // Custom instance data
    rpc_deleter_t           release;    // Callback to release instance data
    rpc_stub_t              stub;       // Client-side stub
};



int _nvm_rpc_handle_insert(nvm_aq_ref ref, uint32_t key, void* data, rpc_deleter_t release)
{
    if (data == NULL || release == NULL)
    {
        return EINVAL;
    }

    struct rpc_handle* handle = (struct rpc_handle*) malloc(sizeof(struct rpc_handle));
    if (handle == NULL)
    {
        dprintf("Failed to allocate binding handle: %s\n", strerror(errno));
        return ENOMEM;
    }

    handle->next = NULL;
    handle->key = key;
    handle->data = data;
    handle->release = release;

    int err = pthread_mutex_lock(&ref->lock);
    if (err != 0)
    {
        free(handle);
        dprintf("Failed to take reference lock: %s\n", strerror(err));
        return err;
    }

    struct rpc_handle* prev = NULL;
    struct rpc_handle* curr = ref->handles;

    while (curr != NULL)
    {
        if (curr->key == handle->key)
        {
            pthread_mutex_unlock(&ref->lock);
            free(handle);
            dprintf("Handle already inserted\n");
            return EINVAL;
        }

        prev = curr;
        curr = curr->next;
    }

    if (prev != NULL)
    {
        prev->next = handle;
    }
    else
    {
        ref->handles = handle;
    }

    ++ref->n_handles;

    pthread_mutex_unlock(&ref->lock);
    return 0;
}



void _nvm_rpc_handle_remove(nvm_aq_ref ref, uint32_t key)
{
    pthread_mutex_lock(&ref->lock);

    struct rpc_handle* prev = NULL;
    struct rpc_handle* curr = ref->handles;

    while (curr != NULL && curr->key != key)
    {
        prev = curr;
        curr = curr->next;
    }

    if (prev != NULL)
    {
        prev->next = curr->next;
    }
    else
    {
        ref->handles = curr->next;
    }

    if (curr != NULL)
    {
        curr->release(curr->data, curr->key, --ref->n_handles);
        free(curr);
    }

    pthread_mutex_unlock(&ref->lock);
}



/*
 * Helper function to remove all server handles.
 * Lock must be held when calling this.
 */
static void release_handles(nvm_aq_ref ref)
{
    struct rpc_handle* curr;
    struct rpc_handle* next;

    if (ref != NULL)
    {
        curr = NULL;
        next = ref->handles;
        while (next != NULL)
        {
            curr = next;
            next = curr->next;
            curr->release(curr->data, curr->key, --ref->n_handles);
            free(curr);
        }

        ref->handles = NULL;
    }
}



/* 
 * Helper function to allocate an admin reference.
 */
int _nvm_ref_get(nvm_aq_ref* handle, const nvm_ctrl_t* ctrl)
{
    *handle = NULL;

    nvm_aq_ref ref = (nvm_aq_ref) malloc(sizeof(struct nvm_admin_reference));
    if (ref == NULL)
    {
        dprintf("Failed to allocate reference: %s\n", strerror(errno));
        return ENOMEM;
    }

    int err = pthread_mutex_init(&ref->lock, NULL);
    if (err != 0)
    {
        free(ref);
        dprintf("Failed to initialize reference lock: %s\n", strerror(err));
        return err;
    }

    ref->ctrl = ctrl;
    ref->handles = NULL;
    ref->data = NULL;
    ref->release = NULL;
    ref->stub = NULL;

    *handle = ref;
    return 0;
}



/* 
 * Helper function to free an admin reference.
 */
void _nvm_ref_put(nvm_aq_ref ref)
{
    if (ref != NULL)
    {
        pthread_mutex_lock(&ref->lock);

        release_handles(ref);

        if (ref->release != NULL)
        {
            ref->release(ref->data, 0, 0);
        }

        pthread_mutex_unlock(&ref->lock);

        pthread_mutex_destroy(&ref->lock);
        free(ref);
    }
}



/* 
 * Execute an NVM admin command.
 * Lock must be held when calling this function.
 */
static int execute_command(struct local_admin* admin, const nvm_cmd_t* cmd, nvm_cpl_t* cpl)
{
    nvm_cmd_t local_copy;
    nvm_cmd_t* in_queue_cmd;
    nvm_cpl_t* in_queue_cpl;

    // Try to enqueue a message
    if ((in_queue_cmd = nvm_sq_enqueue(&admin->asq)) == NULL)
    {
        // Queue was full, but we're holding the lock so no blocking
        return EAGAIN;
    }

    // Copy command into queue slot (but keep original id)
    uint16_t in_queue_id = NVM_DEFAULT_CID(&admin->asq);
    memcpy(&local_copy, cmd, sizeof(nvm_cmd_t));
    *NVM_CMD_CID(&local_copy) = in_queue_id;
    *in_queue_cmd = local_copy;

    // Submit command and wait for completion
    nvm_sq_submit(&admin->asq);

    in_queue_cpl = nvm_cq_dequeue_block(&admin->acq, admin->timeout);
    if (in_queue_cpl == NULL)
    {
        dprintf("Waiting for admin queue completion timed out\n");
        return ETIME;
    }

    nvm_sq_update(&admin->asq);

    // Copy completion and return
    memcpy(cpl, (void*) in_queue_cpl, sizeof(nvm_cpl_t));
    *NVM_CPL_CID(cpl) = *NVM_CMD_CID(cmd);

    return 0;
}



/*
 * Helper function to create a local admin descriptor.
 */
static struct local_admin* create_admin(const nvm_ctrl_t* ctrl, const nvm_dma_t* window)
{
    struct local_admin* admin = (struct local_admin*) malloc(sizeof(struct local_admin));
    
    if (admin == NULL)
    {
        dprintf("Failed to create admin queue-pair descriptors: %s\n", strerror(errno));
        return NULL;
    }

    nvm_queue_clear(&admin->acq, ctrl, true, 0, window->vaddr, window->ioaddrs[0]);

    void* asq_vaddr = (void*) (((unsigned char*) window->vaddr) + window->page_size);
    nvm_queue_clear(&admin->asq, ctrl, false, 0, asq_vaddr, window->ioaddrs[1]);

    memset(window->vaddr, 0, 2 * window->page_size);

    admin->timeout = ctrl->timeout;

    return admin;
}


/*
 * Helper function to remove an admin descriptor.
 */
static void remove_admin(const struct nvm_admin_reference* ref, struct local_admin* admin)
{
    if (ref != NULL)
    {
        free(admin);
    }
}



/*
 * Execute admin command using the RPC binding reference.
 */
int nvm_raw_rpc(nvm_aq_ref ref, nvm_cmd_t* cmd, nvm_cpl_t* cpl)
{
    int err;

    err = pthread_mutex_lock(&ref->lock);
    if (err != 0)
    {
        dprintf("Failed to take reference lock\n");
        return NVM_ERR_PACK(NULL, err);
    }

    if (ref->stub == NULL)
    {
        pthread_mutex_unlock(&ref->lock);
        dprintf("Reference is not bound!\n");
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    err = ref->stub(ref->data, cmd, cpl);

    pthread_mutex_unlock(&ref->lock);

    return NVM_ERR_PACK(cpl, err);
}



/*
 * Bind reference to remote handle.
 */
int _nvm_rpc_bind(nvm_aq_ref ref, void* data, rpc_deleter_t release, rpc_stub_t stub)
{
    int err;

    err = pthread_mutex_lock(&ref->lock);
    if (err != 0)
    {
        dprintf("Failed to take reference lock\n");
        return err;
    }

    if (ref->data != NULL || ref->stub != NULL)
    {
        pthread_mutex_unlock(&ref->lock);
        dprintf("Reference is already bound!\n");
        return EINVAL;
    }

    ref->data = data;
    ref->release = release;
    ref->stub = stub;

    pthread_mutex_unlock(&ref->lock);
    return 0;
}



/*
 * Create admin queues locally.
 */
int nvm_aq_create(nvm_aq_ref* handle, const nvm_ctrl_t* ctrl, const nvm_dma_t* window)
{
    int err;
    nvm_aq_ref ref;

    *handle = NULL;

    if (ctrl->page_size != window->page_size)
    {
        dprintf("Controller page size differs from DMA window page size\n");
        return EINVAL;
    }
    else if (window->n_ioaddrs < 2)
    {
        dprintf("DMA window is not large enough\n");
        return EINVAL;
    }
    else if (window->vaddr == NULL)
    {
        dprintf("DMA window is not mapped into virtual address space\n");
        return EINVAL;
    }

    // Allocate reference
    err = _nvm_ref_get(&ref, ctrl);
    if (err != 0)
    {
        return err;
    }

    // Allocate admin descriptor
    ref->data = create_admin(ref->ctrl, window);
    if (ref->data == NULL)
    {
        _nvm_ref_put(ref);
        return ENOMEM;
    }

    ref->stub = (rpc_stub_t) execute_command;
    ref->release = (rpc_deleter_t) remove_admin;

    // Reset controller
    const struct local_admin* admin = (const struct local_admin*) ref->data;
    nvm_raw_ctrl_reset(ctrl, admin->acq.ioaddr, admin->asq.ioaddr);
    
    *handle = ref;
    return 0;
}



void nvm_aq_destroy(nvm_aq_ref ref)
{
    if (ref != NULL)
    {
        _nvm_ref_put(ref);
    }
}



const nvm_ctrl_t* nvm_ctrl_from_aq_ref(nvm_aq_ref ref)
{
    if (ref != NULL)
    {
        return ref->ctrl;
    }

    return NULL;
}



int _nvm_local_admin(nvm_aq_ref ref, const nvm_cmd_t* cmd, nvm_cpl_t* cpl)
{
    int err = pthread_mutex_lock(&ref->lock);
    if (err != 0)
    {
        dprintf("Failed to take reference lock: %s\n", strerror(err));
        return NVM_ERR_PACK(NULL, err);
    }

    if (ref->stub != (rpc_stub_t) execute_command) 
    {
        pthread_mutex_unlock(&ref->lock);
        dprintf("Reference is not local descriptor\n");
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    err = execute_command((struct local_admin*) ref->data, cmd, cpl);

    pthread_mutex_unlock(&ref->lock);
    return NVM_ERR_PACK(NULL, err);
}



void nvm_rpc_unbind(nvm_aq_ref ref)
{
    if (ref != NULL)
    {
        //if (ref->stub != (rpc_stub_t) execute_command)
        //{
            _nvm_ref_put(ref);
        //}
    }
}

