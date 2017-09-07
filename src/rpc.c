#ifdef _SISCI
#include <sisci_api.h>
#include <sisci_types.h>
#include <sisci_error.h>
#ifndef __DIS_CLUSTER__
#define __DIS_CLUSTER__
#endif
#endif

#include <nvm_types.h>
#include <nvm_rpc.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <string.h>
#include "rpc.h"
#include "dprintf.h"


/* Forward declaration */
struct rpc_reference;



/*
 * RPC reference descriptor.
 *
 * Use this reference to execute NVM commands on a remote (or local) manager.
 */
struct nvm_rpc_reference
{
    struct rpc_reference*       reference;  // Reference to a remote manager.
    nvm_manager_t               manager;    // Reference to a local manager.
};




int nvm_rpc_bind_local(nvm_rpc_t* ref, nvm_manager_t manager)
{
    *ref = NULL;

    nvm_rpc_t handle = (nvm_rpc_t) malloc(sizeof(struct nvm_rpc_reference));
    if (handle == NULL)
    {
        dprintf("Failed to allocate handle: %s\n", strerror(errno));
        return ENOMEM;
    }

    handle->reference = NULL;
    handle->manager = manager;

    *ref = handle;
    return 0;
}



void nvm_rpc_unbind(nvm_rpc_t ref)
{
    if (ref != NULL)
    {
#ifdef _SISCI
        if (ref->reference != NULL)
        {
            _nvm_rpc_ref_free(ref->reference);
            free(ref->reference);
        }
#endif

        free(ref);
    }
}



#ifdef _SISCI

/*
 * Trigger remote interrupt with data.
 */
int _nvm_rpc_ref_send(struct rpc_reference* ref, void* data, size_t length)
{
    sci_error_t err;

    SCITriggerDataInterrupt(ref->intr, data, length, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to trigger data interrupt\n");
        return EIO;
    }

    return 0;
}



int _nvm_rpc_ref_init(struct rpc_reference* ref, uint32_t node_id, uint32_t intr_no, uint32_t adapter)
{
    sci_error_t err;

    ref->node_id = node_id;
    ref->intr_no = intr_no;
    ref->adapter = adapter;

    SCIOpen(&ref->sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to initialize SISCI virtual device: %s\n", SCIGetErrorString(err));
        return EIO;
    }

    SCIConnectDataInterrupt(ref->sd, &ref->intr, node_id, adapter, intr_no, SCI_INFINITE_TIMEOUT, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to connect to remote data interrupt: %s\n", SCIGetErrorString(err));
        SCIClose(ref->sd, 0, &err);
        return EIO;
    }

    return 0;
}



void _nvm_rpc_ref_free(struct rpc_reference* ref)
{
    sci_error_t err;

    do
    {
        SCIDisconnectDataInterrupt(ref->intr, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    SCIClose(ref->sd, 0, &err);
}


#endif /* _SISCI */




#ifdef _SISCI

int nvm_dis_rpc_bind(nvm_rpc_t* ref, uint32_t node_id, uint32_t intr_no, uint32_t adapter)
{
    int err;

    *ref = NULL;

    struct rpc_reference* reference = (struct rpc_reference*) malloc(sizeof(struct rpc_reference));
    if (reference == NULL)
    {
        dprintf("Failed to allocate handle: %s\n", strerror(errno));
        return ENOMEM;
    }

    err = _nvm_rpc_ref_init(reference, node_id, intr_no, adapter);
    if (err != 0)
    {
        free(reference);
        return EIO;
    }

    nvm_rpc_t handle = (nvm_rpc_t) malloc(sizeof(struct nvm_rpc_reference));
    if (handle == NULL)
    {
        free(reference);
        dprintf("Failed to allocate handle: %s\n", strerror(errno));
        return ENOMEM;
    }

    handle->reference = reference;
    handle->manager = NULL;

    *ref = handle;
    return 0;
}

#endif



int nvm_rpc_raw_cmd(nvm_rpc_t ref, const nvm_cmd_t* cmd, nvm_cpl_t* cpl, uint64_t timeout)
{
    int status = EINVAL;
    struct rpc_cmd request;
    struct rpc_cpl reply;

    memcpy(&request.cmd, cmd, sizeof(nvm_cmd_t));

    if (ref->manager != NULL)
    {
        status = _nvm_rpc_local(ref->manager, &request, &reply, timeout);
    }
#ifdef _SISCI
    else if (ref->reference != NULL)
    {
        sci_error_t err;
        struct rpc_handle handle;

        SCIGetLocalNodeId(ref->reference->adapter, &request.node_id, 0, &err);
        if (err != SCI_ERR_OK)
        {
            return EIO;
        }

        status = _nvm_rpc_handle_init(&handle, ref->reference->adapter);
        if (status != 0)
        {
            return status;
        }

        request.intr_no = handle.intr_no;

        status = _nvm_rpc_ref_send(ref->reference, &request, sizeof(request));
        if (status != 0)
        {
            _nvm_rpc_handle_free(&handle);
            return status;
        }

        uint32_t length = sizeof(reply);
        SCIWaitForDataInterrupt(handle.intr, &reply, &length, timeout, 0, &err);
        if (err != SCI_ERR_OK)
        {
            _nvm_rpc_handle_free(&handle);
            dprintf("Did not receive call back from manager\n");
            return EIO;
        }

        _nvm_rpc_handle_free(&handle);

        status = reply.status;
    }
#endif

    memcpy(cpl, &reply.cpl, sizeof(nvm_cpl_t));

    return status;
}

