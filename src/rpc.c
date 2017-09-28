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
#include <nvm_admin.h>
#include <nvm_util.h>
#include <nvm_queue.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <string.h>
#include "regs.h"
#include "rpc.h"
#include "dprintf.h"
#include "dprintnvm.h"


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
    uint64_t                    timeout;    // Controller time out
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
    handle->timeout = 0;

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
        dprintf("Failed to connect to remote data interrupt %u on node %u (%u): %s\n", 
                intr_no, node_id, adapter, SCIGetErrorString(err));
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
    handle->timeout = DIS_CLUSTER_TIMEOUT;

    *ref = handle;
    return 0;
}

#endif



int nvm_rpc_raw_cmd(nvm_rpc_t ref, const nvm_cmd_t* cmd, nvm_cpl_t* cpl)
{
    int status = EINVAL;
    struct rpc_cmd request;
    struct rpc_cpl reply;

    memcpy(&request.cmd, cmd, sizeof(nvm_cmd_t));

    if (ref->manager != NULL)
    {
        status = _nvm_rpc_local(ref->manager, &request, &reply);
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
        SCIWaitForDataInterrupt(handle.intr, &reply, &length, ref->timeout, 0, &err);
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


int nvm_rpc_cq_create(nvm_queue_t* handle, nvm_rpc_t ref, nvm_ctrl_t ctrl, uint16_t id, void* vaddr, uint64_t ioaddr)
{
    nvm_cmd_t command;
    nvm_cpl_t completion;
    nvm_queue_t cq;

    nvm_queue_clear(&cq, ctrl, 1, id, vaddr, ioaddr);

    memset(&command, 0, sizeof(nvm_cmd_t));
    nvm_admin_cq_create(&command, &cq);

    int err = nvm_rpc_raw_cmd(ref, &command, &completion);
    if (err != 0)
    {
        return err;
    }

    if ( ! CPL_OK(&completion) )
    {
        dprintnvm(&command, &completion);
        return EIO;
    }

    *handle = cq;
    return 0;
}


int nvm_rpc_sq_create(nvm_queue_t* handle, nvm_rpc_t ref, nvm_ctrl_t ctrl, const nvm_queue_t* cq, uint16_t id, void* vaddr, uint64_t ioaddr)
{
    nvm_cmd_t command;
    nvm_cpl_t completion;
    nvm_queue_t sq;

    nvm_queue_clear(&sq, ctrl, 0, id, vaddr, ioaddr);

    memset(&command, 0, sizeof(nvm_cmd_t));
    nvm_admin_sq_create(&command, &sq, cq);

    int err = nvm_rpc_raw_cmd(ref, &command, &completion);
    if (err != 0)
    {
        return err;
    }

    if ( ! CPL_OK(&completion) )
    {
        dprintnvm(&command, &completion);
        return EIO;
    }

    *handle = sq;
    return 0;
}


int nvm_rpc_ctrl_info(nvm_ctrl_info_t* info, nvm_rpc_t ref, nvm_ctrl_t ctrl, void* vaddr, uint64_t ioaddr)
{
    int err;
    nvm_cmd_t command;
    nvm_cpl_t completion;

    memset(vaddr, 0, 0x1000);
    memset(info, 0, sizeof(nvm_ctrl_info_t));
    info->nvme_version = (uint32_t) *VER(ctrl->mm_ptr);
    info->page_size = ctrl->page_size;
    info->db_stride = 1UL << ctrl->dstrd;
    info->timeout = ctrl->timeout;
    info->contiguous = !!CAP$CQR(ctrl->mm_ptr);
    info->max_entries = ctrl->max_entries;

    if (vaddr == NULL)
    {
        return 0;
    }

    memset(&command, 0, sizeof(nvm_cmd_t));
    nvm_admin_identify_ctrl(&command, ioaddr);

    err = nvm_rpc_raw_cmd(ref, &command, &completion);
    if (err != 0)
    {
        return err;
    }

    if ( ! CPL_OK(&completion) )
    {
        dprintnvm(&command, &completion);
        return EIO;
    }

    const unsigned char* bytes = ((const unsigned char*) vaddr);

    memcpy(info->pci_vendor, bytes, 4);
    memcpy(info->serial_no, bytes + 4, 20);
    memcpy(info->model_no, bytes + 24, 40);

    info->max_transfer_size = (1UL << bytes[77]) * (1UL << (12 + CAP$MPSMIN(ctrl->mm_ptr)));
    info->sq_entry_size = 1 << _RB(bytes[512], 3, 0);
    info->cq_entry_size = 1 << _RB(bytes[513], 3, 0);
    info->max_out_cmds = *((uint16_t*) (bytes + 514));
    info->max_n_ns = *((uint32_t*) (bytes + 516));

    return 0;
}


int nvm_rpc_ns_info(nvm_ns_info_t* info, nvm_rpc_t ref, uint32_t ns_id, void* vaddr, uint64_t ioaddr)
{
    int err;
    nvm_cmd_t cmd;
    nvm_cpl_t cpl;

    memset(vaddr, 0, 0x1000);
    memset(info, 0, sizeof(nvm_ns_info_t));
    info->ns_id = ns_id;

    if (vaddr == NULL)
    {
        return EINVAL;
    }

    nvm_admin_identify_ns(&cmd, ns_id, ioaddr);

    err = nvm_rpc_raw_cmd(ref, &cmd, &cpl);
    if (err != 0)
    {
        return err;
    }

    if ( ! CPL_OK(&cpl) )
    {
        dprintnvm(&cmd, &cpl);
        return EIO;
    }

    const unsigned char* bytes = (const unsigned char*) vaddr;
    info->size = *((uint64_t*) bytes);
    info->capacity = *((uint64_t*) (bytes + 8));
    info->utilization = *((uint64_t*) (bytes + 16));

    uint8_t format_idx = _RB(bytes[26], 3, 0);

    uint32_t lba_format = *((uint32_t*) (bytes + 128 + sizeof(uint32_t) * format_idx));
    info->lba_data_size = 1 << _RB(lba_format, 23, 16);
    info->metadata_size = _RB(lba_format, 15, 0);

    return 0;
}


int nvm_rpc_get_num_queues(nvm_rpc_t ref, uint16_t* n_cqs, uint16_t* n_sqs)
{
    int err;
    nvm_cmd_t command;
    nvm_cpl_t completion;

    memset(&command, 0, sizeof(nvm_cmd_t));
    nvm_admin_current_num_queues(&command, 0, 0, 0);

    err = nvm_rpc_raw_cmd(ref, &command, &completion);
    if (err != 0)
    {
        return err;
    }

    if ( ! CPL_OK(&completion) )
    {
        dprintnvm(&command, &completion);
        return EIO;
    }

    *n_sqs = (completion.dword[0] >> 16) + 1;
    *n_cqs = (completion.dword[0] & 0xffff) + 1;
    return 0;
}


int nvm_rpc_set_num_queues(nvm_rpc_t ref, uint16_t n_cqs, uint16_t n_sqs)
{
    return nvm_rpc_request_num_queues(ref, &n_cqs, &n_sqs);
}


int nvm_rpc_request_num_queues(nvm_rpc_t ref, uint16_t* n_cqs, uint16_t* n_sqs)
{
    int err;
    nvm_cmd_t command;
    nvm_cpl_t completion;

    if (*n_cqs == 0 || *n_sqs == 0)
    {
        return ERANGE;
    }

    memset(&command, 0, sizeof(nvm_cmd_t));
    nvm_admin_current_num_queues(&command, 1, *n_cqs, *n_sqs);

    err = nvm_rpc_raw_cmd(ref, &command, &completion);
    if (err != 0)
    {
        return err;
    }

    if ( ! CPL_OK(&completion) )
    {
        dprintnvm(&command, &completion);
        return EIO;
    }

    *n_sqs = (completion.dword[0] >> 16) + 1;
    *n_cqs = (completion.dword[0] & 0xffff) + 1;
    return 0;
}

