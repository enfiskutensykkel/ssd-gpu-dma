#ifdef _SISCI
#include <sisci_api.h>
#include <sisci_types.h>
#include <sisci_error.h>
#ifndef __DIS_CLUSTER__
#define __DIS_CLUSTER__
#endif
#endif

#include <nvm_types.h>
#include <nvm_manager.h>
#include <nvm_queue.h>
#include <nvm_ctrl.h>
#include <nvm_command.h>
#include <nvm_admin.h>
#include <nvm_util.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
#include "rpc.h"
#include "util.h"
#include "dprintf.h"


/* Forward declaration */
struct rpc_handle;
struct rpc_list;


/*
 * Controller manager.
 *
 * The manager is responsible for ensuring exclusive access to the admin queues.
 */
struct nvm_manager
{
    pthread_mutex_t     lock;               // Ensure exclusive access to the admin queues
    pthread_cond_t      wait_q;             // Wait queue for queue access
    nvm_queue_t         acq;                // Admin completion queue (ACQ)
    nvm_queue_t         asq;                // Admin submission queue (ASQ)
    uint64_t            ctrl_timeout;       // Controller timeout
    struct rpc_list*    rpc_handlers;       // Linked list of RPC handlers
};



#ifdef _SISCI

static const struct rpc_handle* find_rpc_handle_by_adapter(const nvm_manager_t manager, uint32_t adapter)
{
    const struct rpc_list* ptr;
    
    for (ptr = manager->rpc_handlers; ptr != NULL; ptr = ptr->next)
    {
        if (ptr->handle.adapter == adapter)
        {
            return &ptr->handle;
        }
    }
 
    return NULL;   
}


static const struct rpc_handle* find_rpc_handle_by_intr(const nvm_manager_t manager, sci_local_data_interrupt_t intr)
{
    const struct rpc_list* ptr;
    
    for (ptr = manager->rpc_handlers; ptr != NULL; ptr = ptr->next)
    {
        if (ptr->handle.intr == intr)
        {
            return &ptr->handle;
        }
    }
 
    return NULL;   
}


static sci_callback_action_t handle_rpc(nvm_manager_t manager, sci_local_data_interrupt_t intr, struct rpc_cmd* request, unsigned length, sci_error_t status)
{
    int err;
    struct rpc_reference callback_ref;

    if (status != SCI_ERR_OK)
    {
        dprintf("Unexpected status in handler routine: %s\n", SCIGetErrorString(status));
        return SCI_CALLBACK_CANCEL;
    }

    if (length != sizeof(struct rpc_cmd))
    {
        dprintf("Received unknown data in handler routine\n");
        return SCI_CALLBACK_CONTINUE;
    }

    // Find local handle
    const struct rpc_handle* handle = find_rpc_handle_by_intr(manager, intr);
    if (handle == NULL)
    {
        dprintf("RPC handle not found in handler routine\n");
        return SCI_CALLBACK_CANCEL;
    }

    // Create a remote reference so we can call back
    err = _nvm_rpc_ref_init(&callback_ref, request->node_id, request->intr_no, handle->adapter);
    if (err != 0)
    {
        dprintf("Failed to create callback reference in handler routine\n");
        return SCI_CALLBACK_CONTINUE;
    }
    
    // Create a reply
    struct rpc_cpl reply;
    reply.status = 0;
    reply.accepted = false;
    memset(&reply.cpl, 0, sizeof(nvm_cpl_t));

    // Check if the command is accepted
    if ( handle->filter == NULL || handle->filter( request->node_id, handle->adapter, handle->intr_no, (nvm_cmd_t*) &request->cmd ) )
    {
        // It was, execute it...
        _nvm_rpc_local(manager, request, &reply, manager->ctrl_timeout);
    }

    // Call back with the response
    _nvm_rpc_ref_send(&callback_ref, &reply, sizeof(struct rpc_cpl));

    _nvm_rpc_ref_free(&callback_ref);

    return SCI_CALLBACK_CONTINUE;
}



static int initialize_rpc_handle(nvm_manager_t manager, struct rpc_handle* handle, uint32_t intr_no, uint32_t adapter, nvm_rpc_filter_t filter)
{
    sci_error_t err;

    handle->intr_no = intr_no;
    handle->adapter = adapter;
    handle->filter = filter;

    SCIOpen(&handle->sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to initialize SISCI virtual device: %s\n", SCIGetErrorString(err));
        return EIO;
    }

    uint32_t flags = 0;
    sci_cb_data_interrupt_t callback = NULL;

    if (manager != NULL)
    {
        flags = SCI_FLAG_USE_CALLBACK | SCI_FLAG_FIXED_INTNO;
        callback = (sci_cb_data_interrupt_t) handle_rpc;
    }

    SCICreateDataInterrupt(handle->sd, &handle->intr, adapter, &handle->intr_no, callback, manager, flags, &err);

    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to create local data interrupt: %s\n", SCIGetErrorString(err));
        SCIClose(handle->sd, 0, &err);
        return EIO;
    }

    return 0;
}



int _nvm_rpc_handle_init(struct rpc_handle* handle, uint32_t adapter)
{
    return initialize_rpc_handle(NULL, handle, 0, adapter, NULL);
}



void _nvm_rpc_handle_free(struct rpc_handle* handle)
{
    sci_error_t err;

    do
    {
        SCIRemoveDataInterrupt(handle->intr, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    SCIClose(handle->sd, 0, &err);
}



int nvm_dis_rpc_enable(nvm_manager_t manager, uint32_t adapter, uint32_t intr_no, nvm_rpc_filter_t filter)
{
    int err;
    struct rpc_list* insert;

    if (find_rpc_handle_by_adapter(manager, adapter) != NULL)
    {
        return EINVAL;
    }
   
    insert = (struct rpc_list*) malloc(sizeof(struct rpc_list));
    if (insert == NULL)
    {
        dprintf("Failed to allocate RPC manager handle: %s\n", strerror(errno));
        return ENOMEM;
    }

    insert->next = NULL;

    if ((err = pthread_mutex_lock(&manager->lock)) != 0)
    {
        free(insert);
        dprintf("Failed to acquire manager lock: %s\n", strerror(err));
        return err;
    }

    err = initialize_rpc_handle(manager, &insert->handle, intr_no, adapter, filter);
    if (err != 0)
    {
        free(insert);
        pthread_mutex_unlock(&manager->lock);
        return err;
    }

    if (manager->rpc_handlers == NULL)
    {
        manager->rpc_handlers = insert;
    }
    else
    {
        struct rpc_list* ptr;
        for (ptr = manager->rpc_handlers; ptr->next != NULL; ptr = ptr->next);
        ptr->next = insert;        
    }

    pthread_mutex_unlock(&manager->lock);
    return 0;
}



int nvm_dis_rpc_disable(nvm_manager_t manager, uint32_t adapter)
{
    int err;

    if ((err = pthread_mutex_lock(&manager->lock)) != 0)
    {
        dprintf("Failed to acquire manager lock: %s\n", strerror(err));
        return err;
    }

    struct rpc_list* prev = NULL;
    struct rpc_list* curr = manager->rpc_handlers;

    while (curr != NULL)
    {
        if (curr->handle.adapter == adapter)
        {
            break;
        }

        prev = curr;
        curr = curr->next;
    }

    err = EINVAL;
    if (curr != NULL)
    {
        if (prev != NULL)
        {
            manager->rpc_handlers = curr->next;
        }
        else
        {
            prev->next = curr->next;
        }

        _nvm_rpc_handle_free(&curr->handle);
        free(curr);
        err = 0;
    }

    pthread_mutex_unlock(&manager->lock);
    return err;
}

#endif /* _SISCI */




/*
 * Execute an NVM admin command.
 */
int _nvm_rpc_local(nvm_manager_t manager, const struct rpc_cmd* cmd, struct rpc_cpl* cpl, uint64_t timeout)
{
    int err;
    nvm_cmd_t* queue_cmd;
    nvm_cpl_t* queue_cpl;

    cpl->accepted = false;
    cpl->status = 0;

    if ((err = pthread_mutex_lock(&manager->lock)) != 0)
    {
        return err;
    }
    
    // Try to enqueue a message
    while ((queue_cmd = sq_enqueue(&manager->asq)) == NULL)
    {
        // Queue is full, wait until a later time
        if ((err = pthread_cond_wait(&manager->wait_q, &manager->lock)) != 0)
        {
            pthread_mutex_unlock(&manager->lock);
            cpl->status = err;
            return err;
        }
    }
    
    // Copy command into queue slot (but keep original command ID)
    uint16_t queue_cmd_id = *CMD_CID(queue_cmd);
    memcpy(queue_cmd, &cmd->cmd, sizeof(nvm_cmd_t));
    *CMD_CID(queue_cmd) = queue_cmd_id;

    // Submit command and wait for reply
    sq_submit(&manager->asq);
    
    queue_cpl = cq_dequeue_block(&manager->acq, timeout);
    if (queue_cpl == NULL)
    {
        dprintf("Waiting for admin completion timed out\n");
        pthread_mutex_unlock(&manager->lock);
        cpl->status = ETIME;
        return ETIME;
    }

    // Copy completion and update queue tail and head
    sq_update(&manager->asq, queue_cpl);
    pthread_cond_signal(&manager->wait_q);
    pthread_mutex_unlock(&manager->lock);

    memcpy(&cpl->cpl, queue_cpl, sizeof(nvm_cpl_t));
    *CPL_CID(&cpl->cpl) = *CMD_CID(&cmd->cmd);
    cq_update(&manager->acq);

    cpl->accepted = true;
    cpl->status = 0;
    return 0;
}



/*
 * Register admin queue manager.
 */
int nvm_manager_register(nvm_manager_t* handle, nvm_ctrl_t ctrl, nvm_dma_t window)
{
    int err;
    nvm_manager_t manager;

    *handle = NULL;

    // Do some sanity checking
    if (window->n_ioaddrs < 2)
    {
        dprintf("Queue memory is not large enough\n");
        return ERANGE;
    }
    else if (ctrl->page_size != window->page_size)
    {
        dprintf("DMA window page size mismatch\n");
        return EINVAL;
    }
    else if (window->vaddr == NULL)
    {
        dprintf("Virtual queue memory is not mapped\n");
        return EINVAL;
    }

    // Allocate handle struct
    manager = (nvm_manager_t) malloc(sizeof(struct nvm_manager));
    if (manager == NULL)
    {
        dprintf("Failed to allocate manager handle: %s\n", strerror(errno));
        return ENOMEM;
    }

    // Initialize lock and wait queue
    err = pthread_mutex_init(&manager->lock, NULL);
    if (err != 0)
    {
        dprintf("Failed to initialize manager lock: %s\n", strerror(err));
        goto free_handle;
    }

    err = pthread_cond_init(&manager->wait_q, NULL);
    if (err != 0)
    {
        dprintf("Failed to initialize wait queue: %s\n", strerror(err));
        goto destroy_mtx;
    }

    // Create queue handlers
    nvm_queue_clear(&manager->acq, ctrl, true, 0, window->vaddr, window->ioaddrs[0]);

    void* asq_ptr = (void*) (((unsigned char*) window->vaddr) + window->page_size);
    nvm_queue_clear(&manager->asq, ctrl, false, 0, asq_ptr, window->ioaddrs[1]);
    
    memset(window->vaddr, 0, 2 * window->page_size);

    err = nvm_ctrl_reset(ctrl, manager->acq.ioaddr, manager->asq.ioaddr);
    if (err != 0)
    {
        dprintf("Failed to reset controller");
        goto destroy_cv;
    }

    manager->ctrl_timeout = ctrl->timeout;
    manager->rpc_handlers = NULL;

    *handle = manager;
    return 0;

destroy_cv:
    pthread_cond_destroy(&manager->wait_q);

destroy_mtx:
    pthread_mutex_destroy(&manager->lock);

free_handle:
    free(manager);

    return err;
}



/*
 * Destroy admin queue manager.
 */
void nvm_manager_unregister(nvm_manager_t manager)
{
    if (manager != NULL)
    {

#ifdef _SISCI
        pthread_mutex_lock(&manager->lock);
        struct rpc_list* ptr = manager->rpc_handlers;

        while (ptr != NULL)
        {
            manager->rpc_handlers = manager->rpc_handlers->next;

            _nvm_rpc_handle_free(&ptr->handle);
            free(ptr);

            ptr = manager->rpc_handlers;
        }
        pthread_mutex_unlock(&manager->lock);
#endif

        pthread_cond_destroy(&manager->wait_q);
        pthread_mutex_destroy(&manager->lock);
        free(manager);
    }
}

