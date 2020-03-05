#ifndef _SISCI
#error "Must compile with SISCI support"
#endif

#ifndef __DIS_CLUSTER__
#define __DIS_CLUSTER__
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <errno.h>
#include "dis/interrupt.h"
#include "dprintf.h"
#include <sisci_types.h>
#include <sisci_error.h>
#include <sisci_api.h>



/*
 * Do some sanity checking and then call supplied callback.
 */ 
static sci_callback_action_t interrupt_callback(struct local_intr* interrupt, 
                                                sci_local_data_interrupt_t intr,
                                                void* data,
                                                uint32_t length,
                                                sci_error_t status)
{
#ifndef NDEBUG
    if (status != SCI_ERR_OK)
    {
        dprintf("Unexpected status in interrupt handler routine: %s\n", _SCIGetErrorString(status));
        return SCI_CALLBACK_CANCEL;
    }

    if (intr != interrupt->intr)
    {
        dprintf("Possible memory corruption\n");
        return SCI_CALLBACK_CANCEL;
    }
#endif

    interrupt->callback(interrupt->data, data, length);

    return SCI_CALLBACK_CONTINUE;
}



int _nvm_local_intr_get(struct local_intr* intr, uint32_t adapter, void* cb_data, intr_callback_t cb)
{
    sci_error_t err = SCI_ERR_OK;

    // Get local node identifier
    SCIGetLocalNodeId(adapter, &intr->node_id, 0, &err);
#ifndef NDEBUG
    if (err != SCI_ERR_OK)
    {
        dprintf("Unexpected error: %s\n", _SCIGetErrorString(err));
        return EIO;
    }
#endif

    // Open SISCI descriptor
    SCIOpen(&intr->sd, 0, &err);
#ifndef NDEBUG
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to open SISCI virtual device: %s\n", _SCIGetErrorString(err));
        return EIO;
    }
#endif

    intr->adapter = adapter;
    intr->data = cb_data;
    intr->callback = cb;
    
    uint32_t flags = 0;
    void* data = NULL;
    sci_cb_data_interrupt_t callback = NULL;

    // Callback was supplied, set up parameters
    if (cb != NULL)
    {
        data = (void*) intr;
        callback = (sci_cb_data_interrupt_t) interrupt_callback;
        flags |= SCI_FLAG_USE_CALLBACK;
    }

    // Create data interrupt
    SCICreateDataInterrupt(intr->sd, &intr->intr, adapter, &intr->intr_no, callback, data, flags, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to create data interrupt: %s\n", _SCIGetErrorString(err));
        SCIClose(intr->sd, 0, &err);
        return ENOSPC;
    }

    return 0;
}



void _nvm_local_intr_put(struct local_intr* intr)
{
    sci_error_t err = SCI_ERR_OK;

    do
    {
        SCIRemoveDataInterrupt(intr->intr, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    SCIClose(intr->sd, 0, &err);
}



int _nvm_local_intr_wait(struct local_intr* intr, void* data, uint16_t maxlen, uint32_t timeout)
{
    sci_error_t err = SCI_ERR_OK;
    uint32_t len = maxlen;
    
    SCIWaitForDataInterrupt(intr->intr, data, &len, timeout, 0, &err);

    switch (err)
    {
        case SCI_ERR_OK:
            return 0;

        case SCI_ERR_TIMEOUT:
            return ETIMEDOUT;

        default:
            dprintf("Waiting for data interrupt unexpectedly failed: %s\n", _SCIGetErrorString(err));
            return EIO;
    }
}



int _nvm_remote_intr_get(struct remote_intr* intr, uint32_t adapter, uint32_t node, uint32_t no)
{
    sci_error_t err = SCI_ERR_OK;

    SCIOpen(&intr->sd, 0, &err);
#ifndef NDEBUG
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to open SISCI virtual device: %s\n", _SCIGetErrorString(err));
        return EIO;
    }
#endif

    SCIConnectDataInterrupt(intr->sd, &intr->intr, node, adapter, no, SCI_INFINITE_TIMEOUT, 0, &err);
    if (err != SCI_ERR_OK)
    {
        SCIClose(intr->sd, 0, &err);
        return ECONNREFUSED;
    }

    return 0;
}



void _nvm_remote_intr_put(struct remote_intr* intr)
{
    sci_error_t err = SCI_ERR_OK;
    SCIDisconnectDataInterrupt(intr->intr, 0, &err);
    SCIClose(intr->sd, 0, &err);
}



/*
 * Trigger remote interrupt with data.
 */
int _nvm_remote_intr_trigger(const struct remote_intr* intr, void* data, uint16_t length)
{
    sci_error_t err = SCI_ERR_OK;

    SCITriggerDataInterrupt(intr->intr, data, length, 0, &err);
    if (err != SCI_ERR_OK)
    {
        dprintf("Failed to trigger data interrupt\n");
        return ENOTCONN;
    }

    return 0;
}



/*
 * Convenience function for easy remote interrupt triggering.
 */
int _nvm_remote_intr_fire_and_forget(uint32_t adapter, uint32_t node, uint32_t no, void* data, uint16_t len)
{
    int status = 0;
    struct remote_intr intr;

    status = _nvm_remote_intr_get(&intr, adapter, node, no);
    if (status != 0)
    {
        return status;
    }

    status = _nvm_remote_intr_trigger(&intr, data, len);
    _nvm_remote_intr_put(&intr);
    return status;
}

