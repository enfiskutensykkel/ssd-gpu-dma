#include <nvm_types.h>
#include <nvm_manager.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_ctrl.h>
#include <nvm_admin.h>
#include <nvm_command.h>
#include "pcidev.h"
#include "regs.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <sys/time.h>
#include <time.h>
#include <pthread.h>

#ifdef _SISCI
#include <sisci_api.h>
#include <sisci_types.h>
#include <sisci_error.h>
#endif

#ifndef NDEBUG
#include <stdio.h>
#endif


#define SCIGetErrorString(x) SCIGetErrorStr(x) // FIXME: Remove this when using updated version of SISCI


#ifdef _SISCI
/* Format of interrupt data */
struct command_request
{
    uint32_t                    node_id;    // Local node identifier so manager can "call" back
    uint32_t                    intno;      // Local interrupt number so manager can "call" back
    nvm_cmd_t                   cmd;        // The admin command we want to execute remotely
};


/* NVM manager export descriptor */
struct export
{
    struct export*              next;       // Next pointer in the export list
    sci_desc_t                  sd;         // SISCI virtual device (for data interrupt)
    uint32_t                    adapter;    // DIS adapter number
    uint32_t                    intno;      // Interrupt number used for export
    sci_local_data_interrupt_t  intr;       // Local data interrupt
};

/* NVM manager memory descriptor */
struct memory
{
    sci_desc_t          sd;                 // SISCI virtual device (for segment)
    uint32_t            segment_id;         // Unique segment identifier
    sci_local_segment_t segment;            // Local SISCI segment
    sci_ioaddr_t        ioaddr;             // IO address of segment on far side of NTB
};
#endif /* _SISCI */


/* NVM manager descriptor */
struct nvm_manager
{
    const nvm_ctrl_t*   ctrl;               // NVM controller reference
    pthread_mutex_t     lk;                 // Ensure exclusive access to controller admin queues
    pthread_cond_t      asq_wq;             // Wait queue for ASQ access
    pthread_cond_t      acq_wq;             // Wait queue for ACQ access
    nvm_queue_t         acq;                // Admin completion queue (ACQ)
    nvm_queue_t         asq;                // Admin submission queue (ASQ)
    void*               vaddr;              // Queue virtual memory pointer
    uint64_t            ioaddr[3];          // Bus addresses of memory
#ifdef _SISCI
    struct memory       memory;             // Segment memory description
    sci_map_t           map;                // Memory map handle 
    size_t              n_exports;          // Number of exports
    struct export*      exports_head;       // Linked list of exports
#endif /* _SISCI */
};


/* 
 * Helper function to enqueue an admin command.
 *
 * Lock must be held when entering this function.
 */
static nvm_cmd_t* enqueue_block(nvm_mngr_t mngr, const nvm_cmd_t* command)
{
    nvm_cmd_t* cmd;

    while ((cmd = sq_enqueue(&mngr->asq)) == NULL)
    {
        // Queue is full, wait until a later time
        pthread_cond_wait(&mngr->asq_wq, &mngr->lk);
    }
    
    uint16_t orig_cmd_id = *CMD_CID(cmd);
    memcpy(cmd, command, sizeof(nvm_cmd_t));
    *CMD_CID(cmd) = orig_cmd_id;

    sq_submit(&mngr->asq);

    return cmd;
}


/* 
 * Helper function to dequeue a admin completion.
 *
 * Lock must be held when entering this function.
 */
static int dequeue_block(nvm_mngr_t mngr, nvm_cpl_t* completion, const nvm_cmd_t* cmd, uint64_t timeout)
{
    nvm_cpl_t* cpl;

    while ((cpl = cq_poll(&mngr->acq)) == NULL || *CPL_CID(cpl) != *CMD_CID(cmd))
    {
        if (--timeout == 0)
        {
            return ETIME;
        }
        
        if (cpl != NULL)
        {
            pthread_cond_broadcast(&mngr->acq_wq);
        }

        struct timeval now;
        gettimeofday(&now, NULL);

        uint64_t secs = now.tv_sec + (now.tv_usec + 1000UL) / 1000000UL;
        uint64_t usecs = (now.tv_usec + 1000UL) % 1000000UL;

        struct timespec abstime;
        abstime.tv_sec = secs;
        abstime.tv_nsec = usecs * 1000UL;

        pthread_cond_timedwait(&mngr->acq_wq, &mngr->lk, &abstime);
    }

    nvm_cpl_t* dequeued_cpl = cq_dequeue(&mngr->acq);
#ifndef NDEBUG
    if (dequeued_cpl != cpl)
    {
        fprintf(stderr, "Errornous state -- dequeued completion is not expected completion\n");
        return EBADFD;
    }
#endif

    sq_update(&mngr->asq, cpl);
    pthread_cond_broadcast(&mngr->asq_wq);

    memcpy(completion, cpl, sizeof(nvm_cpl_t));
    cq_update(&mngr->acq);

    return 0;
}


/* Helper function to set up admin queues and reset controller */
static int reset_controller(nvm_mngr_t mngr)
{
    void* acq_ptr = mngr->vaddr;
    uint64_t acq_addr = mngr->ioaddr[0];
    void* asq_ptr = (void*) (((unsigned char*) mngr->vaddr) + mngr->ctrl->page_size);
    uint64_t asq_addr = mngr->ioaddr[1];

    nvm_queue_clear(&mngr->acq, mngr->ctrl, 1, 0, acq_ptr, acq_addr);
    nvm_queue_clear(&mngr->asq, mngr->ctrl, 0, 0, asq_ptr, asq_addr);

    memset(mngr->vaddr, 0, mngr->ctrl->page_size * 3);

    return nvm_ctrl_reset(mngr->ctrl, mngr->acq.ioaddr, mngr->asq.ioaddr);
}


int nvm_ctrl_get_info(nvm_mngr_t mngr, nvm_ctrl_info_t* info)
{
    nvm_cpl_t cpl;

    info->nvme_version = (uint32_t) *VER(mngr->ctrl->mm_ptr);

    uint64_t ioaddr = mngr->ioaddr[2];

    nvm_cmd_t identify;
    memset(&identify, 0, sizeof(nvm_cmd_t));
    nvm_cmd_header(&identify, NVM_ADMIN_IDENTIFY_CONTROLLER, 0);
    nvm_cmd_data_ptr(&identify, ioaddr, 0);
    identify.dword[10] = (0 << 16) | 0x01;

    nvm_cmd_t features;
    memset(&features, 0, sizeof(nvm_cmd_t));
    nvm_cmd_header(&features, NVM_ADMIN_GET_FEATURES, 0);
    nvm_cmd_data_ptr(&features, 0, 0);
    features.dword[10] = (0x03 << 8) | 0x07;
    features.dword[11] = 0;

    int err = nvm_local_cmd(mngr, &identify, &cpl, mngr->ctrl->timeout);
    if (err != 0)
    {
        return err;
    }

    if (SCT(&cpl) != 0 && SC(&cpl) != 0)
    {
#ifndef NDEBUG
        fprintf(stderr, "IDENTIFY CONTROLLER command failed with status code type %u and status code %u: %s\n", 
                SCT(&cpl), SC(&cpl), nvm_strerror(&cpl));
#endif
        return EIO;
    }

    unsigned char* bytes = ((unsigned char*) mngr->vaddr) + 2 * mngr->ctrl->page_size;

    memcpy(info->pci_vendor, bytes, 4);
    memcpy(&info->serial_no, bytes + 4, 20);
    memcpy(&info->model_no, bytes + 24, 40);

    fprintf(stderr, "So far so good\n");

    info->max_data_size = bytes[77] * (1 << (12 + CAP$MPSMIN(mngr->ctrl->mm_ptr)));
    info->sq_entry_size = 1 << _RB(bytes[512], 3, 0);
    info->cq_entry_size = 1 << _RB(bytes[513], 3, 0);
    info->max_out_cmds = *((uint16_t*) (bytes + 514));
    info->n_ns = *((uint32_t*) (bytes + 516));

    err = nvm_local_cmd(mngr, &features, &cpl, mngr->ctrl->timeout);
    if (err != 0)
    {
        return err;
    }

    if (SCT(&cpl) != 0 && SC(&cpl) != 0)
    {
#ifndef NDEBUG
        fprintf(stderr, "GET FEATURES command failed with status code type %u and status code %u: %s\n", 
                SCT(&cpl), SC(&cpl), nvm_strerror(&cpl));
#endif
        return EIO;
    }

    info->max_sqs = (cpl.dword[0] >> 16) + 1;
    info->max_cqs = (cpl.dword[0] & 0xffff) + 1;

    return 0;
}


int nvm_local_cmd(nvm_mngr_t mngr, const struct nvm_command* cmd, struct nvm_completion* cpl, uint64_t timeout)
{
    int err = 0;
    nvm_cmd_t* command;

    pthread_mutex_lock(&mngr->lk);
    command = enqueue_block(mngr, cmd);
    err = dequeue_block(mngr, cpl, command, timeout);
    pthread_mutex_unlock(&mngr->lk);

#ifndef NDEBUG
    if (err != 0)
    {
        fprintf(stderr, "Waiting for admin command completion timedout\n");
    }
#endif

    return err;
}


#ifdef _SISCI

/* Helper function to look up adapter number from a local interrupt */
static struct export* find_export_intr(nvm_mngr_t mngr, sci_local_data_interrupt_t intr)
{
    struct export* ptr = mngr->exports_head;

    while (ptr != NULL)
    {
        if (ptr->intr == intr)
        {
            return ptr;
        }

        ptr = ptr->next;
    }

    return NULL;
}


/* Helper function to look up an export from an adapter number */
static struct export* find_export_adapter(nvm_mngr_t mngr, uint32_t adapter, struct export** prev)
{
    *prev = NULL;
    struct export* ptr = mngr->exports_head;

    while (ptr != NULL)
    {
        if (ptr->adapter == adapter)
        {
            return ptr;
        }

        *prev = ptr;
        ptr = ptr->next;
    }

    return NULL;
}


/* Helper function to insert an export in the exports linked list */
static void insert_export(nvm_mngr_t mngr, struct export* prev, struct export* export)
{
    if (prev == NULL)
    {
        mngr->exports_head = export;
    }
    else
    {
        prev->next = export;
    }
}


/* Helper function to remove an export from the exports linked list */
static void remove_export(nvm_mngr_t mngr, struct export* prev, struct export* export)
{
    if (prev == NULL)
    {
        mngr->exports_head = export->next;
    }
    else
    {
        prev->next = export->next;
    }
}


/* Callback routine for handling remote NVM admin command requests */
static sci_callback_action_t handle_command_request(
        nvm_mngr_t mngr, 
        sci_local_data_interrupt_t intr, 
        struct command_request* cmdreq,
        unsigned int length,
        sci_error_t status
        )
{
    sci_error_t err;

    if (status != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Unexpected status in local data interrupt handler: %s\n", SCIGetErrorString(status));
#endif
        return SCI_CALLBACK_CANCEL;
    }

    struct export* export = find_export_intr(mngr, intr);
    if (export == NULL)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to look up export on local data interrupt\n");
#endif
        return SCI_CALLBACK_CANCEL;
    }

    unsigned int local_node_id = 0;
    SCIGetLocalNodeId(export->adapter, &local_node_id, 0, &err);

    if (length != sizeof(struct command_request))
    {
#ifndef NDEBUG
        fprintf(stderr, "Received unknown data in data interrupt %u on node %u adapter %u\n", 
                export->intno, local_node_id, export->adapter);
#endif
        return SCI_CALLBACK_CONTINUE;
    }

    pthread_mutex_lock(&mngr->lk);

    nvm_cmd_t* cmd = enqueue_block(mngr, &cmdreq->cmd);

    nvm_cpl_t cpl;
    if (dequeue_block(mngr, &cpl, cmd, mngr->ctrl->timeout) != 0)
    {
        pthread_mutex_unlock(&mngr->lk);
#ifndef NDEBUG
        fprintf(stderr, "Waiting for admin command completion timedout\n");
#endif
        return SCI_CALLBACK_CONTINUE;
    }

    pthread_mutex_unlock(&mngr->lk);

    // Return completion to remote command initiator
    nvm_remote_cpl(cmdreq->node_id, export->adapter, cmdreq->node_id, &cpl, mngr->ctrl->timeout / 2);

    return SCI_CALLBACK_CONTINUE;
}


/* Helper function for triggering a remote data interrupt */
static sci_error_t trigger_interrupt(sci_desc_t sd, uint32_t node_id, uint32_t adapter, uint32_t intno, void* data, size_t len, uint32_t timeout)
{
    sci_error_t err;
    sci_error_t status;

    sci_remote_data_interrupt_t interrupt;
    SCIConnectDataInterrupt(sd, &interrupt, node_id, adapter, intno, SCI_INFINITE_TIMEOUT, 0, &err); // TODO: wait for timeout instead
    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to connect to data interrupt %u on node %u: %x\n", intno, node_id, err);
#endif
        return err;
    }

    SCITriggerDataInterrupt(interrupt, data, len, 0, &err);
    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to trigger data interrupt %u on node %u: %x\n", intno, node_id, err);
#endif
        status = err;
        SCIDisconnectDataInterrupt(interrupt, 0, &err);
        return status;
    }

    SCIDisconnectDataInterrupt(interrupt, 0, &err);
    return err;
}


int nvm_remote_cmd(uint32_t remote_node_id, uint32_t adapter, uint32_t remote_intno, const nvm_cmd_t* cmd, nvm_cpl_t* cpl, uint64_t timeout)
{
    sci_error_t err;
    int status = 0;

    uint32_t local_node_id;
    SCIGetLocalNodeId(adapter, &local_node_id, 0, &err);
    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Unexpected error when retrieving local node identifier: %s\n", SCIGetErrorString(err));
#endif
        return EIO;
    }

    sci_desc_t sd;
    SCIOpen(&sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Unexpected error when creating handle: %s\n", SCIGetErrorString(err));
#endif
        return ENOSPC;
    }

    sci_local_data_interrupt_t intr;
    unsigned local_intno = remote_intno; // FIXME: Data interrupts should work without SCI_FLAG_FIXED_INTO soon

    SCICreateDataInterrupt(sd, &intr, adapter, &local_intno, NULL, NULL, SCI_FLAG_FIXED_INTNO /* FIXME: remove this */, &err);
    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to create local data interrupt: %s\n", SCIGetErrorString(err));
#endif
        status = EIO;
        goto quit;
    }

    struct command_request cmdreq;
    cmdreq.node_id = local_node_id;
    cmdreq.intno = local_intno;
    memcpy(&cmdreq.cmd, cmd, sizeof(nvm_cmd_t));

    status = trigger_interrupt(sd, remote_node_id, adapter, remote_intno, &cmdreq, sizeof(struct command_request), timeout / 2);
    if (status != 0)
    {
        goto remove;
    }

    unsigned recvd_len = sizeof(nvm_cpl_t);
    SCIWaitForDataInterrupt(intr, cpl, &recvd_len, timeout / 2, 0, &err);
    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Waiting for admin completion timed out: %s\n", SCIGetErrorString(err));
#endif
        status = ETIME;
        goto remove;
    }

remove:
    do
    {
        SCIRemoveDataInterrupt(intr, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

quit:
    SCIClose(sd, 0, &err);
    return status;
}


int nvm_remote_cpl(uint32_t node_id, uint32_t adapter, uint32_t intno, const nvm_cpl_t* cpl, uint64_t timeout)
{
    sci_error_t err;
    int status = 0;

    sci_desc_t sd;
    SCIOpen(&sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Unexpected error when creating handle: %s\n", SCIGetErrorString(err));
#endif
        return ENOSPC;
    }

    if (cpl != NULL)
    {
        nvm_cpl_t reply;
        memcpy(&reply, cpl, sizeof(nvm_cpl_t));

        status = trigger_interrupt(sd, node_id, adapter, intno, &reply, sizeof(nvm_cpl_t), timeout);
    }
    else
    {
        status = trigger_interrupt(sd, node_id, adapter, intno, NULL, 0, timeout);
    }

    SCIClose(sd, 0, &err);
    return status;
}


static void destroy_queue_memory(const nvm_ctrl_t* ctrl, struct memory* mem)
{
    sci_error_t err;

    do
    {
        SCIUnmapSegmentForDevice(mem->segment, ctrl->dev_ref->device, ctrl->dev_ref->adapter, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    do
    {
        SCISetSegmentUnavailable(mem->segment, ctrl->dev_ref->adapter, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    do
    {
        SCIRemoveSegment(mem->segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    SCIClose(mem->sd, 0, &err);
}


static int create_queue_memory(const nvm_ctrl_t* ctrl, struct memory* mem, uint32_t segment_id)
{
    sci_error_t err;
    int status;

    if (ctrl->dev_ref == NULL)
    {
        return EINVAL;
    }

    SCIOpen(&mem->sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to create SISCI virtual device: %s\n", SCIGetErrorString(err));
#endif
        return ENOSPC;
    }

    SCICreateSegment(mem->sd, &mem->segment, segment_id, ctrl->page_size * 3, NULL, NULL, 0, &err);
    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to create queue memory: %s\n", SCIGetErrorString(err));
#endif
        status = ENOMEM;
        goto close;
    }

    SCIPrepareSegment(mem->segment, ctrl->dev_ref->adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to export queue memory for controller: %s\n", SCIGetErrorString(err));
#endif
        status = EIO;
        goto unmap;
    }

    SCISetSegmentAvailable(mem->segment, ctrl->dev_ref->adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to export queue memory for controller: %s\n", SCIGetErrorString(err));
#endif
        status = EIO;
        goto unmap;
    }

    SCIMapSegmentForDevice(mem->segment,ctrl->dev_ref->device, ctrl->dev_ref->adapter, &mem->ioaddr, 0, &err);
    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to create DMA window to queue memory from controller: %s\n", SCIGetErrorString(err));
#endif
        status = EIO;
        goto set_unavailable;
    }

    return 0;

set_unavailable:
    do
    {
        SCISetSegmentUnavailable(mngr->segment, mngr->ctrl->dev_ref->adapter, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

remove:
    do
    {
        SCIRemoveSegment(mngr->segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

close:
    SCIClose(mngr->sd, 0, &err);

    return status;
}


int nvm_mngr_export(nvm_mngr_t mngr, uint32_t adapter, uint32_t intno)
{
    sci_error_t err;

    struct export* prev = NULL;
    struct export* export = find_export_adapter(mngr, adapter, &prev);
    if (export != NULL)
    {
#ifndef NDEBUG
        fprintf(stderr, "Controller manager is already exported on adapter %u\n", adapter);
#endif
        return EINVAL;
    }

    export = (struct export*) malloc(sizeof(struct export));
    if (export == NULL)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to allocate manager export descriptor: %s\n", strerror(errno));
#endif
        return errno;
    }

    export->next = NULL;
    export->adapter = adapter;
    export->intno = intno;
    
    SCIOpen(&export->sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        free(export);
#ifndef NDEBUG
        fprintf(stderr, "Failed to create SISCI virtual device: %s\n", SCIGetErrorString(err));
#endif
        return ENOSPC;
    }

    SCICreateDataInterrupt(export->sd, &export->intr, export->adapter, &export->intno, (sci_cb_data_interrupt_t) handle_command_request, mngr, SCI_FLAG_USE_CALLBACK | SCI_FLAG_FIXED_INTNO, &err);
    if (err != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to create remote command handler: %s\n", SCIGetErrorString(err));
#endif
        SCIClose(export->sd, 0, &err);
        free(export);
        return EIO;
    }
    
    insert_export(mngr, prev, export);
    return 0;
}


int nvm_mngr_unexport(nvm_mngr_t mngr, uint32_t adapter)
{
    sci_error_t err;

    struct export* prev = NULL;
    struct export* export = find_export_adapter(mngr, adapter, &prev);
    if (export == NULL)
    {
#ifndef NDEBUG
        fprintf(stderr, "Controller manager is not exported on adapter %u\n", adapter);
#endif
        return EINVAL;
    }

    do
    {
        SCIRemoveDataInterrupt(export->intr, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

#ifndef NDEBUG
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to remove remote command handler: %s\n", SCIGetErrorString(err));
    }
#endif

    SCIClose(export->sd, 0, &err);
    
    remove_export(mngr, prev, export);
    free(export);

    return 0;
}


static void remove_all_exports(struct export* exports)
{
    while (exports != NULL)
    {
        sci_error_t err;
        struct export* next;

        do
        {
            SCIRemoveDataInterrupt(exports->intr, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

#ifndef NDEBUG
        if (err != SCI_ERR_OK)
        {
            fprintf(stderr, "Failed to remove remote command handler: %s\n", SCIGetErrorString(err));
        }
#endif

        SCIClose(exports->sd, 0, &err);

        next = exports->next;
        free(exports);
        exports = next;
    }
}


void sisci_free(nvm_mngr_t manager)
{
    sci_error_t err;

    if (manager->ctrl->dev_ref == NULL)
    {
        return;
    }

    remove_all_exports(manager->exports_head, manager->n_exports);

    do
    {
        SCIUnmapSegment(manager->map, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    destroy_queue_memory(manager->ctrl, &manager->memory);
}


int nvm_mngr_init(nvm_mngr_t* handle, const nvm_ctrl_t* ctrl, uint32_t segment_id)
{
    int err;

    if (ctrl->dev_ref == NULL)
    {
        return EINVAL;
    }

    struct memory segment_mem;
    err = create_queue_memory(ctrl, &segment_mem, segment_id);
    if (err != 0)
    {
        return err;
    }

    sci_map_t map;
    sci_error_t status;
    void* vaddr = SCIMapLocalSegment(segment_mem.segment, &map, 0, 3 * ctrl->page_size, NULL, 0, &status);
    if (status != SCI_ERR_OK)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to memory map queue memory: %s\n", SCIGetErrorString(status));
#endif
        return EIO;
    }

    uint64_t ioaddrs[3];
    ioaddr[0] = (uint64_t) segment_mem.ioaddr;
    ioaddr[1] = ((uint64_t) segment_mem.ioaddr) + ctrl->page_size;
    ioaddr[2] = ((uint64_t) segment_mem.ioaddr) + 2 * ctrl->page_size;
    
    memset(vaddr, 0, 3 * ctrl->page_size);
    err = nvm_mngr_init_raw(handle, ctrl, vaddr, ioaddrs);
    if (err != 0)
    {
        do
        {
            SCIUnmapSegment(map, 0, &status);
        }
        while (status == SCI_ERR_BUSY);
    }

    manager->memory = segment_mem;
    manager->map = map;
    manager->n_exports = 0;
    manager->exports_head = NULL;

    *handle = manager;
    return 0;
}

#endif /* #ifdef _SISCI */


int nvm_mngr_init_raw(nvm_mngr_t* handle, const nvm_ctrl_t* ctrl, void* vaddr, size_t n_ioaddrs, const uint64_t* ioaddrs)
{
    int err;

    if (n_ioaddrs < 3)
    {
        return ERANGE;
    }

    nvm_mngr_t manager = (nvm_mngr_t) malloc(sizeof(struct nvm_manager));
    if (manager == NULL)
    {
#ifndef NDEBUG
        fprintf(stderr, "Failed to allocate NVM manager handle: %s\n", strerror(errno));
#endif
        return errno;
    }

    manager->ctrl = ctrl;

    err = pthread_mutex_init(&manager->lk, NULL);
    if (err != 0)
    {
        goto free_handle;
    }

    err = pthread_cond_init(&manager->asq_wq, NULL);
    if (err != 0)
    {
        goto destroy_mtx;
    }

    err = pthread_cond_init(&manager->acq_wq, NULL);
    if (err != 0)
    {
        goto destroy_asq_wq;
    }

    manager->vaddr = vaddr;
    manager->ioaddr[0] = ioaddrs[0];
    manager->ioaddr[1] = ioaddrs[1];
    manager->ioaddr[2] = ioaddrs[2];

    err = reset_controller(manager);
    if (err != 0)
    {
        goto destroy_acq_wq;
    }

    *handle = manager;
    return 0;

destroy_acq_wq:

destroy_asq_wq:
    pthread_cond_destroy(&manager->asq_wq);

destroy_mtx:
    pthread_mutex_destroy(&manager->lk);

free_handle:
    free(manager);

    return err;
}


void nvm_mngr_free(nvm_mngr_t manager)
{
    if (manager != NULL)
    {
#ifdef _SISCI
        sisci_free(manager);
#endif
        pthread_cond_destroy(&manager->acq_wq);
        pthread_cond_destroy(&manager->asq_wq);
        pthread_mutex_destroy(&manager->lk);
        free(manager);
    }
}

