#ifndef _SISCI
#error "Must compile with SISCI support"
#endif

#ifndef __DIS_CLUSTER__
#define __DIS_CLUSTER__
#endif

#include <nvm_types.h>
#include <nvm_rpc.h>
#include <nvm_aq.h>
#include <nvm_util.h>
#include <nvm_queue.h>
#include <nvm_error.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <string.h>
#include "dis/device.h"
#include "dis/map.h"
#include "dis/interrupt.h"
#include "rpc.h"
#include "regs.h"
#include "ctrl.h"
#include "dprintf.h"
#include <sisci_types.h>
#include <sisci_error.h>
#include <sisci_api.h>
#include <dis/dis_types.h>


#define RPC_COMMAND_TIMEOUT     2500
#define RPC_MAGIC_SIGNATURE     0xDEADBEEF
#define N_ADAPTERS              DIS_MAX_NSCIS


/*
 * RPC command message format.
 */
struct __attribute__((packed)) rpc_cmd
{
    uint32_t                    node_id;    // Initiator node identifier
    uint32_t                    intr_no;    // Callback interrupt number
    unsigned char               cmd[64];    // Command to execute
};



/*
 * RPC completion message format.
 */
struct __attribute__((packed)) rpc_cpl
{
    unsigned char               cmd[64];    // Modified command (zero'd means rejected)
    unsigned char               cpl[16];    // Command completion
};



/*
 * Information about binding handle (exported on shared device memory)
 */
struct __attribute__((packed)) handle_info
{
    uint32_t                    magic;      // Magic signature
    uint32_t                    node_id;    // Node identifier
    uint32_t                    intr_no;    // Interrupt number
};



/*
 * Local RPC binding handle.
 */
struct binding_handle
{
    struct controller*          ctrl;       // Controller reference
    sci_remote_segment_t        segment;    // Device memory segment
    nvm_aq_ref                  rpc_ref;    // RPC reference
    nvm_dis_rpc_cb_t            rpc_cb;     // RPC callback
    struct local_intr           intr;       // Interrupt handle
};



/*
 * Remote RPC binding.
 */
struct binding
{
    sci_remote_segment_t        segment;    // Device memory segment
    struct local_intr           lintr;      // Local interrupt handle
    struct remote_intr          rintr;      // Remote interrupt handle
};



/*
 * Handle remote command request.
 */
static void handle_remote_command(struct binding_handle* handle, struct rpc_cmd* request, uint16_t len)
{
    struct rpc_cpl reply;

    uint32_t adapter = handle->intr.adapter;
    uint32_t node_id = request->node_id;
    uint32_t intr_no = request->intr_no;

    // Sanity checking
    if (len != sizeof(struct rpc_cmd))
    {
        dprintf("Got unexpected data in RPC binding handle\n");
        return;
    }
    
    // Allow user callback to modify request in place
    if ( handle->rpc_cb == NULL || handle->rpc_cb((nvm_cmd_t*) &request->cmd, adapter, node_id) )
    {
        _nvm_local_admin(handle->rpc_ref, (const nvm_cmd_t*) &request->cmd, (nvm_cpl_t*) &reply.cpl);
        memcpy(&reply.cmd, &request->cmd, sizeof(nvm_cmd_t));
    }
    else
    {
        memset(&reply.cmd, 0, sizeof(nvm_cmd_t));
    }

    // Send modified command and completion back
    int status = _nvm_remote_intr_fire_and_forget(adapter, node_id, intr_no, &reply, sizeof(reply));
    if (status != 0)
    {
        dprintf("Failed to establish reverse connection: %s\n", strerror(errno));
    }
}



/*
 * Initiate remote command request.
 */
static int remote_command(struct binding* binding, nvm_cmd_t* cmd, nvm_cpl_t* cpl)
{
    struct rpc_cmd request;
    struct rpc_cpl reply;

    if (cmd == NULL || cpl == NULL)
    {
        return NVM_ERR_PACK(NULL, EINVAL);
    }

    request.node_id = binding->lintr.node_id;
    request.intr_no = binding->lintr.intr_no;
    memcpy(&request.cmd, cmd, sizeof(nvm_cmd_t));

    // Trigger remote interrupt
    int status = _nvm_remote_intr_trigger(&binding->rintr, &request, sizeof(request));
    if (status != 0)
    {
        return NVM_ERR_PACK(NULL, status);
    }

    // XXX: Can a race condition occur here? 
    // XXX: Maybe create interrupt with callback instead and wait for cond var here?
    
    // Wait for callback interrupt
    status = _nvm_local_intr_wait(&binding->lintr, &reply, sizeof(reply), RPC_COMMAND_TIMEOUT);
    if (status != 0)
    {
        return NVM_ERR_PACK(NULL, status);
    }

    memcpy(cmd, &reply.cmd, sizeof(nvm_cmd_t));
    memcpy(cpl, &reply.cpl, sizeof(nvm_cpl_t));

    return NVM_ERR_PACK(NULL, 0);
}



/* 
 * Helper function to write info about a handle.
 */
static int write_handle_info(const struct binding_handle* handle, uint32_t adapter, bool clear)
{
    sci_error_t err;
    sci_map_t map;
    const size_t size = sizeof(struct handle_info) * N_ADAPTERS;
    volatile struct handle_info* infos;

    infos = (volatile struct handle_info*) SCIMapRemoteSegment(handle->segment, &map, 0, size, NULL, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to map shared device segment: %s\n", _SCIGetErrorString(err));
        return EIO;
    }

    struct handle_info info;
    info.magic = clear ? 0 : RPC_MAGIC_SIGNATURE;
    info.node_id = clear ? 0 : handle->intr.node_id;
    info.intr_no = clear ? 0 : handle->intr.intr_no;

    infos[adapter] = info;

    nvm_wcb_flush();

    do
    {
        SCIUnmapSegment(map, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    return 0;
}



/*
 * Create shared segment for writing connection information and connect to it.
 */
static int try_create(sci_remote_segment_t* segment, const struct device* dev)
{
    int status;
    sci_remote_segment_t rseg;
    unsigned adapter = 0;
    
    *segment = NULL;

    // Try to connect to segment (and create it if that fails)
    status = _nvm_device_memory_get(&rseg, dev, 0, SCI_MEMTYPE_SHARED);
    if (status != 0)
    {
        sci_error_t err;

        SCICreateDeviceSegment(dev->device, 0, sizeof(struct handle_info) * N_ADAPTERS, 
                SCI_MEMTYPE_SHARED, SCI_MEMACCESS_HOST_WRITE | SCI_MEMACCESS_MULTIHOST_READ, 0, &err);

        switch (err)
        {
            case SCI_ERR_SEGMENTID_USED:
            case SCI_ERR_OK:
                break;

            default:
                dprintf("Unexpected error when creating shared device segment: %s\n", _SCIGetErrorString(err));
                return EIO;
        }

        status = _nvm_device_memory_get(&rseg, dev, 0, SCI_MEMTYPE_SHARED);
        if (status != 0)
        {
            return status;
        }
    }

    // Check if we are the owner of the segment
    for (adapter = 0; adapter < N_ADAPTERS; ++adapter)
    {
        sci_error_t err;
        uint32_t node_id = 0;

        SCIGetLocalNodeId((unsigned int) adapter, &node_id, 0, &err);
        if (err == SCI_ERR_OK)
        {
            if (SCIGetRemoteSegmentNodeId(rseg) == node_id)
            {
                break;
            }
        }
    }

    if (adapter == N_ADAPTERS)
    {
        return EEXIST;
    }

    *segment = rseg;
    return 0;
}   



/*
 * Helper function to create a a server binding handle.
 */
static int create_binding_handle(struct binding_handle** handle, nvm_aq_ref ref, uint32_t adapter, nvm_dis_rpc_cb_t cb)
{
    int status;
    *handle = NULL;
    sci_remote_segment_t rseg;

    const nvm_ctrl_t* ctrl = nvm_ctrl_from_aq_ref(ref);
    if (ctrl == NULL)
    {
        return EINVAL;
    }

    struct controller* cref = _nvm_ctrl_get(ctrl);
    if (cref == NULL)
    {
        return ENOTTY;
    }

    status = try_create(&rseg, cref->device);
    if (status != 0)
    {
        _nvm_ctrl_put(cref);
        return status;
    }

    struct binding_handle* bh = (struct binding_handle*) malloc(sizeof(struct binding_handle));
    if (bh == NULL)
    {
        _nvm_device_memory_put(&rseg);
        _nvm_ctrl_put(cref);
        dprintf("Failed to allocate RPC binding handle: %s\n", strerror(errno));
        return ENOMEM;
    }

    bh->ctrl = cref;
    bh->segment = rseg;
    bh->rpc_ref = ref;
    bh->rpc_cb = cb;
    
    status = _nvm_local_intr_get(&bh->intr, adapter, bh, (intr_callback_t) handle_remote_command);
    if (status != 0)
    {
        _nvm_device_memory_put(&bh->segment);
        _nvm_ctrl_put(bh->ctrl);
        free(bh);
        return status;
    }

    status = write_handle_info(bh, adapter, false);
    if (status != 0)
    {
        _nvm_device_memory_put(&bh->segment);
        _nvm_ctrl_put(bh->ctrl);
        free(bh);
        return status;
    }

    *handle = bh;
    return 0;
}



/*
 * Helper function to remove a server binding handle.
 */
static void remove_binding_handle(uint32_t adapter, struct binding_handle* handle)
{
    write_handle_info(handle, adapter, true);

    _nvm_local_intr_put(&handle->intr);
    _nvm_device_memory_put(&handle->segment);
    _nvm_ctrl_put(handle->ctrl);

    free(handle);
}



/*
 * Helper function to try to connect to remote interrupt.
 */
static int try_bind(struct binding* binding, uint32_t adapter, unsigned max)
{
    sci_error_t err;
    sci_map_t map;
    volatile const struct handle_info* info;

    // Create mapping to remote shared segment
    info = SCIMapRemoteSegment(binding->segment, &map, 0, sizeof(struct handle_info) * max, NULL, SCI_FLAG_READONLY_MAP, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to map remote segment: %s\n", _SCIGetErrorString(err));
        return EIO;
    }

    // Iterate over exported interrupts
    for (unsigned i = 0; i < max; ++i)
    {
        // Read information
        uint32_t magic = info[i].magic;
        uint32_t node_id = info[i].node_id;
        uint32_t intr_no = info[i].intr_no;

        if (magic != RPC_MAGIC_SIGNATURE)
        {
            continue;
        }

        // Attempt to connect
        if (_nvm_remote_intr_get(&binding->rintr, adapter, node_id, intr_no) == 0)
        {
            SCIUnmapSegment(map, 0, &err);
            return 0;
        }
    }

    SCIUnmapSegment(map, 0, &err);
    //dprintf("Failed to connect to remote interrupt\n");
    return ECONNREFUSED;
}


/*
 * Helper function to connect to device shared memory,
 * extract handle info and connect to interrupt.
 */
static int create_binding(struct binding** handle, const struct device* dev, uint32_t adapter)
{
    int status;
    *handle = NULL;

    struct binding* binding = (struct binding*) malloc(sizeof(struct binding));
    if (binding == NULL)
    {
        dprintf("Failed to allocate binding descriptor: %s\n", strerror(errno));
        return ENOMEM;
    }

    status = _nvm_device_memory_get(&binding->segment, dev, 0, SCI_MEMTYPE_SHARED);
    if (status != 0)
    {
        free(binding);
        return status; // Assume that device memory segment isn't allocated
    }

    status = _nvm_local_intr_get(&binding->lintr, adapter, NULL, NULL);
    if (status != 0)
    {
        _nvm_device_memory_put(&binding->segment);
        free(binding);
        return status;
    }

    status = try_bind(binding, adapter, N_ADAPTERS);
    if (status != 0)
    {
        _nvm_local_intr_put(&binding->lintr);
        _nvm_device_memory_put(&binding->segment);
        free(binding);
        return status;
    }

    *handle = binding;
    return 0;
}



/*
 * Helper function to disconnect from remote interrupt and 
 * shared device memory.
 */
static void remove_binding(struct binding* binding)
{
    _nvm_remote_intr_put(&binding->rintr);
    _nvm_local_intr_put(&binding->lintr);
    _nvm_device_memory_put(&binding->segment);

    free(binding);
}



int nvm_dis_rpc_enable(nvm_aq_ref ref, uint32_t adapter, nvm_dis_rpc_cb_t filter)
{
    int err;
    struct binding_handle* handle;

    if (adapter >= N_ADAPTERS)
    {
        return EINVAL;
    }

    err = create_binding_handle(&handle, ref, adapter, filter);
    if (err != 0)
    {
        return err;
    }

    err = _nvm_rpc_handle_insert(ref, adapter, handle, (rpc_free_handle_t) remove_binding_handle);
    if (err != 0)
    {
        remove_binding_handle(adapter, handle);
        return err;
    }
    
    return 0;
}



void nvm_dis_rpc_disable(nvm_aq_ref ref, uint32_t adapter)
{
    _nvm_rpc_handle_remove(ref, adapter);
}



int nvm_dis_rpc_bind(nvm_aq_ref* handle, const nvm_ctrl_t* ctrl, uint32_t adapter)
{
    nvm_aq_ref ref;
    *handle = NULL;

    int err = _nvm_ref_get(&ref, ctrl);
    if (err != 0)
    {
        return err;
    }

    struct binding* binding;
    err = create_binding(&binding, _nvm_container_of(ctrl, struct controller, handle)->device, adapter);
    if (err != 0)
    {
        _nvm_ref_put(ref);
        return err;
    }

    err = _nvm_rpc_bind(ref, binding, (rpc_free_binding_t) remove_binding, (rpc_stub_t) remote_command);
    if (err != 0)
    {
        remove_binding(binding);
        _nvm_ref_put(ref);
        return err;
    }

    *handle = ref;
    return 0;
}

