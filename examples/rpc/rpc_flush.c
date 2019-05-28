#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_admin.h>
#include <nvm_dma.h>
#include <nvm_rpc.h>
#include <nvm_aq.h>
#include <nvm_util.h>
#include <nvm_error.h>
#include <nvm_queue.h>
#include <nvm_cmd.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <getopt.h>
#include <string.h>
#include <errno.h>
#include <sisci_api.h>
#include "segment.h"
#include "util.h"
#include <unistd.h>


static int flush(nvm_queue_t* cq, nvm_queue_t* sq, uint32_t ns)
{
    nvm_cmd_t cmd;
    memset(&cmd, 0, sizeof(cmd));
    nvm_cpl_t* cpl;

    nvm_cmd_header(&cmd, NVM_DEFAULT_CID(sq), NVM_IO_FLUSH, ns);

    nvm_cmd_t* ptr = nvm_sq_enqueue(sq);
    if (ptr == NULL)
    {
        return ENOMEM;
    }

    *ptr = cmd;

    nvm_sq_submit(sq);

    while ((cpl = nvm_cq_dequeue(cq)) == NULL)
    {
        //usleep(1);
    }

    nvm_sq_update(sq);
    nvm_cq_update(cq);

    int status = NVM_ERR_STATUS(cpl);
    fprintf(stderr, "%s\n", nvm_strerror(status));

    return status;
}


int main(int argc, char** argv)
{
    nvm_ctrl_t* ctrl;
    nvm_aq_ref rpc;
    sci_error_t scierr;
    struct segment segment;
    nvm_dma_t* dma;
    nvm_queue_t cq;
    nvm_queue_t sq;

    SCIInitialize(0, &scierr);
    if (scierr != SCI_ERR_OK)
    {
        exit(1);
    }

    int err = nvm_dis_ctrl_init(&ctrl, 0xc0c00);

    err = nvm_dis_rpc_bind(&rpc, ctrl, 0);
    
    err = segment_create(&segment, 123, 0x2000);

    err = dma_create(&dma, ctrl, &segment, 0);

    err = nvm_admin_cq_create(rpc, &cq, 1, dma, 0, 2);

    err = nvm_admin_sq_create(rpc, &sq, &cq, 1, dma, 1, 2);

    fprintf(stderr, "OK\n");

    for (int i = 0; i < 10; ++i)
    {
        flush(&cq, &sq, 1);
    }

    nvm_admin_sq_delete(rpc, &sq, &cq);
    nvm_admin_cq_delete(rpc, &cq);
    dma_remove(dma, &segment, 0);
    segment_remove(&segment);
    nvm_rpc_unbind(rpc);
    nvm_ctrl_free(ctrl);


    SCITerminate();
    return 0;
}
