#include "message.h"
#include "nvm/command.h"
#include <stddef.h>
#include <stdint.h>
#include <sisci_api.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>


static sci_error_t trigger_interrupt(sci_desc_t sd, uint32_t node_id, uint32_t intno, void* data, size_t len, uint32_t timeout)
{
    sci_error_t err;
    sci_error_t status;

    sci_remote_data_interrupt_t interrupt;
    SCIConnectDataInterrupt(sd, &interrupt, node_id, 0, intno, 0, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to connect to interrupt %u on node %u: %x\n", intno, node_id, err);
        return err;
    }

    SCITriggerDataInterrupt(interrupt, data, len, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to trigger interrupt %u on node %u: %x\n", intno, node_id, err);
        status = err;
        SCIDisconnectDataInterrupt(interrupt, 0, &err);
        return status;
    }

    SCIDisconnectDataInterrupt(interrupt, 0, &err);
    return err;
}


int send_completion(uint32_t node_id, uint32_t intno, const struct completion* cpl, uint32_t timeout)
{
    sci_error_t err;
    sci_desc_t sd;
    int status;

    SCIOpen(&sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to open SCI virtual device: %x\n", err);
        return EIO;
    }

    if (cpl != NULL)
    {
        struct completion copy;
        memcpy(&copy, cpl, sizeof(copy));

        status = trigger_interrupt(sd, node_id, intno, &copy, sizeof(copy), timeout);
    }
    else
    {
        status = trigger_interrupt(sd, node_id, intno, NULL, 0, timeout);
    }

    if (status != 0)
    {
        fprintf(stderr, "Failed to trigger data interrupt: %s\n", strerror(status));
    }

    SCIClose(sd, 0, &err);
    return status;
}


int remote_command(uint32_t remote_node_id, uint32_t remote_intno, const struct command* cmd, struct completion* cpl, uint32_t timeout)
{
    sci_error_t err;
    int status;

    uint32_t local_node_id;
    SCIGetLocalNodeId(0, &local_node_id, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to retrieve local node id: %x\n", err);
        return EIO;
    }

    sci_desc_t sd;
    SCIOpen(&sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to open SCI virtual device: %x\n", err);
        return EIO;
    }

    sci_local_data_interrupt_t interrupt;
    unsigned local_intno = remote_intno; // FIXME: Remove SCI_FLAG_FIXED_INTNO

    SCICreateDataInterrupt(sd, &interrupt, 0, &local_intno, NULL, NULL, SCI_FLAG_FIXED_INTNO /* TODO: Remove this flag */, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to create data interrupt: %x\n", err);
        status = EIO;
        goto end;
    }

    struct command copy;
    memcpy(&copy, cmd, sizeof(copy));
    copy.dword[14] = local_node_id;
    copy.dword[15] = local_intno;

    status = trigger_interrupt(sd, remote_node_id, remote_intno, &copy, sizeof(copy), timeout);
    if (status != 0)
    {
        fprintf(stderr, "Failed to trigger data interrupt: %s\n", strerror(status));
        goto remove;
    }

    unsigned recvd_len = sizeof(struct completion);
    SCIWaitForDataInterrupt(interrupt, cpl, &recvd_len, 0, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Waiting for completion timed out: %x\n", err);
        status = ETIME;
        goto remove;
    }

    status = 0;

remove:
    do
    {
        SCIRemoveDataInterrupt(interrupt, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

end:
    SCIClose(sd, 0, &err);
    return status;
}
