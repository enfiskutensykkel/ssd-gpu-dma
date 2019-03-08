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
#include <nvm_dma.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include "mutex.h"
#include "rpc.h"
#include "ctrl.h"
#include "util.h"
#include "dprintf.h"



/*
 * Local admin queue-pair descriptor.
 */
struct local_admin
{
    nvm_dma_t*          qmem;       // Queue memory
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
    rpc_free_handle_t   release;    // Callback to release the instance data
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
    struct controller*      ctrl;       // Controller reference
    struct mutex            lock;       // Ensure exclusive access to the reference
    struct rpc_handle*      handles;    // Linked list of binding handles (if server)
    void*                   data;       // Custom instance data
    rpc_free_binding_t      release;    // Callback to release instance data
    rpc_stub_t              stub;       // Client-side stub
};



int _nvm_rpc_handle_insert(nvm_aq_ref ref, uint32_t key, void* data, rpc_free_handle_t release)
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

    int err = _nvm_mutex_lock(&ref->lock);
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
            _nvm_mutex_unlock(&ref->lock);
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

    _nvm_mutex_unlock(&ref->lock);
    return 0;
}



void _nvm_rpc_handle_remove(nvm_aq_ref ref, uint32_t key)
{
    _nvm_mutex_lock(&ref->lock);

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
        curr->release(curr->key, curr->data);
        free(curr);
    }

    _nvm_mutex_unlock(&ref->lock);
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
            curr->release(curr->key, curr->data);
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
    struct controller* cref;

    cref = _nvm_ctrl_get(ctrl);
    if (cref == NULL)
    {
        return ENOTTY;
    }

    nvm_aq_ref ref = (nvm_aq_ref) malloc(sizeof(struct nvm_admin_reference));
    if (ref == NULL)
    {
        _nvm_ctrl_put(cref);
        dprintf("Failed to allocate reference: %s\n", strerror(errno));
        return ENOMEM;
    }

    int err = _nvm_mutex_init(&ref->lock);
    if (err != 0)
    {
        _nvm_ctrl_put(cref);
        free(ref);
        dprintf("Failed to initialize reference lock: %s\n", strerror(err));
        return err;
    }

    ref->ctrl = cref;
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
        _nvm_mutex_lock(&ref->lock);

        release_handles(ref);

        if (ref->release != NULL)
        {
            ref->release(ref->data);
        }
        
        _nvm_ctrl_put(ref->ctrl);
        _nvm_mutex_unlock(&ref->lock);

        _nvm_mutex_free(&ref->lock);
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
    *cpl = *in_queue_cpl;
    nvm_cq_update(&admin->acq);

    *NVM_CPL_CID(cpl) = *NVM_CMD_CID(cmd);

    return 0;
}



/*
 * Helper function to create a local admin descriptor.
 */
static int create_admin(struct local_admin** handle, const struct controller* ctrl, const nvm_dma_t* window)
{
    int status;
    nvm_dma_t* copy;
    struct local_admin* admin;

    *handle = NULL;

    if (ctrl->handle.page_size != window->page_size)
    {
        dprintf("Controller page size differs from DMA window page size\n");
        return EINVAL;
    }
    else if (window->n_ioaddrs < 2)
    {
        dprintf("DMA window is not large enough\n");
        return ERANGE;
    }
    else if (window->vaddr == NULL)
    {
        dprintf("DMA window is not mapped into virtual address space\n");
        return EINVAL;
    }

    status = nvm_dma_remap(&copy, window);
    if (status != 0)
    {
        return status;
    }

    admin = (struct local_admin*) malloc(sizeof(struct local_admin));
    if (admin == NULL)
    {
        nvm_dma_unmap(copy);
        dprintf("Failed to create admin queue-pair descriptors: %s\n", strerror(errno));
        return errno;
    }

    admin->qmem = copy;
    memset((void*) admin->qmem->vaddr, 0, 2 * admin->qmem->page_size);

    nvm_queue_clear(&admin->acq, &ctrl->handle, true, 0, ctrl->handle.page_size / sizeof(nvm_cpl_t), 
            admin->qmem->local, admin->qmem->vaddr, admin->qmem->ioaddrs[0]);

    nvm_queue_clear(&admin->asq, &ctrl->handle, false, 0, ctrl->handle.page_size / sizeof(nvm_cmd_t), 
            admin->qmem->local,  NVM_DMA_OFFSET(admin->qmem, 1), admin->qmem->ioaddrs[1]);

    admin->timeout = ctrl->handle.timeout;

    *handle = admin;
    return 0;
}


/*
 * Helper function to remove an admin descriptor.
 */
static void remove_admin(struct local_admin* admin)
{
    if (admin != NULL)
    {
        nvm_dma_unmap(admin->qmem);
        free(admin);
    }
}



/*
 * Execute admin command using the RPC binding reference.
 */
int nvm_raw_rpc(nvm_aq_ref ref, nvm_cmd_t* cmd, nvm_cpl_t* cpl)
{
    int err;

    err = _nvm_mutex_lock(&ref->lock);
    if (err != 0)
    {
        dprintf("Failed to take reference lock\n");
        return NVM_ERR_PACK(NULL, err);
    }

    if (ref->stub == NULL)
    {
        _nvm_mutex_unlock(&ref->lock);
        dprintf("Reference is not bound!\n");
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    err = ref->stub(ref->data, cmd, cpl);

    _nvm_mutex_unlock(&ref->lock);

    return NVM_ERR_PACK(cpl, err);
}



/*
 * Bind reference to remote handle.
 */
int _nvm_rpc_bind(nvm_aq_ref ref, void* data, rpc_free_binding_t release, rpc_stub_t stub)
{
    int err;

    err = _nvm_mutex_lock(&ref->lock);
    if (err != 0)
    {
        dprintf("Failed to take reference lock\n");
        return err;
    }

    if (ref->data != NULL || ref->stub != NULL)
    {
        _nvm_mutex_unlock(&ref->lock);
        dprintf("Reference is already bound!\n");
        return EINVAL;
    }

    ref->data = data;
    ref->release = release;
    ref->stub = stub;

    _nvm_mutex_unlock(&ref->lock);
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

    // Allocate reference
    err = _nvm_ref_get(&ref, ctrl);
    if (err != 0)
    {
        return err;
    }

    // Allocate admin descriptor
    err = create_admin((struct local_admin**) &ref->data, ref->ctrl, window);
    if (err != 0)
    {
        _nvm_ref_put(ref);
        return err;
    }

    ref->stub = (rpc_stub_t) execute_command;
    ref->release = (rpc_free_binding_t) &remove_admin;

    // Reset controller
    const struct local_admin* admin = (const struct local_admin*) ref->data;
    nvm_raw_ctrl_reset(ctrl, admin->qmem->ioaddrs[0], admin->qmem->ioaddrs[1]);
    
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
        return &ref->ctrl->handle;
    }

    return NULL;
}



int _nvm_local_admin(nvm_aq_ref ref, const nvm_cmd_t* cmd, nvm_cpl_t* cpl)
{
    int err = _nvm_mutex_lock(&ref->lock);
    if (err != 0)
    {
        dprintf("Failed to take reference lock: %s\n", strerror(err));
        return NVM_ERR_PACK(NULL, err);
    }

    if (ref->stub != (rpc_stub_t) execute_command) 
    {
        _nvm_mutex_unlock(&ref->lock);
        dprintf("Reference is not local descriptor\n");
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    err = execute_command((struct local_admin*) ref->data, cmd, cpl);

    _nvm_mutex_unlock(&ref->lock);
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

