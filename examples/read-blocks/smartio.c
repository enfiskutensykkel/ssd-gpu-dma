#include "args.h"
#include "read.h"
#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_dma.h>
#include <nvm_aq.h>
#include <nvm_admin.h>
#include <nvm_util.h>
#include <nvm_error.h>
#include <sisci_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>


int main(int argc, char** argv)
{
    int status;
    sci_error_t err;

    struct disk_info info;
    struct queue_pair queues;

    nvm_ctrl_t* ctrl = NULL;
    nvm_dma_t* aq_mem = NULL;
    nvm_aq_ref aq_ref = NULL;
    nvm_dma_t* buffer = NULL;
    nvm_dma_t* cq_mem = NULL;
    nvm_dma_t* sq_mem = NULL;

    struct options args;

    parse_options(argc, argv, &args);

    SCIInitialize(0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Something went wrong: %s\n", SCIGetErrorString(err));
        exit(1);
    }
    
    // Get controller reference
    status = nvm_dis_ctrl_init(&ctrl, args.controller_id);
    if (!nvm_ok(status))
    {
        fprintf(stderr, "Failed to get controller reference: %s\n", nvm_strerror(status));
        goto leave;
    }

    // Create admin queue pair + page for identify commands
    status = nvm_dis_dma_create(&aq_mem, ctrl, ctrl->page_size * 3, 0);
    if (!nvm_ok(status))
    {
        fprintf(stderr, "Failed to create queue memory: %s\n", nvm_strerror(status));
        goto leave;
    }

    // Reset controller and set admin queue pair
    status = nvm_aq_create(&aq_ref, ctrl, aq_mem);
    if (!nvm_ok(status))
    {
        fprintf(stderr, "Failed to reset controller: %s\n", nvm_strerror(status));
        goto leave;
    }

    // Identify controller and namespace
    status = get_disk_info(aq_ref, &info, args.namespace_id, NVM_DMA_OFFSET(aq_mem, 2), aq_mem->ioaddrs[2], args.identify);
    if (status != 0)
    {
        goto leave;
    }

    // Create data buffer
    size_t buffer_size = (args.chunk_size <= args.num_blocks ? args.chunk_size : args.num_blocks) * info.block_size;

    status = nvm_dis_dma_create(&buffer, ctrl, buffer_size, 0);
    if (!nvm_ok(status))
    {
        fprintf(stderr, "Failed to create data buffer: %s\n", nvm_strerror(status));
        goto leave;
    }

    // Create memory for completion queue
    status = nvm_dis_dma_create(&cq_mem, ctrl, ctrl->page_size, SCI_MEMACCESS_HOST_READ | SCI_MEMACCESS_DEVICE_WRITE);
    if (!nvm_ok(status))
    {
        fprintf(stderr, "Failed to create completion queue memory: %s\n", nvm_strerror(status));
        goto leave;
    }

    // Create memory for submission queue and PRP lists
    size_t n_prp_lists = args.queue_size;

    status = nvm_dis_dma_create(&sq_mem, ctrl, 
            NVM_SQ_PAGES(ctrl, args.queue_size) * ctrl->page_size + ctrl->page_size * (n_prp_lists + 1), 
            SCI_MEMACCESS_HOST_WRITE | SCI_MEMACCESS_DEVICE_READ); 
    if (!nvm_ok(status))
    {
        goto leave;
    }

    // Create queues
    status = create_queue_pair(aq_ref, &queues, cq_mem, sq_mem, args.queue_size);
    if (!nvm_ok(status))
    {
        goto leave;
    }

    if (args.input != NULL)
    {
        status = write_blocks(&info, &queues, buffer, &args);
        if (status != 0)
        {
            goto leave;
        }
    }

    status = read_and_dump(&info, &queues, buffer, &args);

    
leave:
    if (args.input != NULL)
    {
        fclose(args.input);
    }

    if (args.output != NULL)
    {
        fprintf(stderr, "Flushing output file...\n");
        fclose(args.output);
    }
    fprintf(stderr, "Done\n");
    nvm_dma_unmap(cq_mem);
    nvm_dma_unmap(sq_mem);
    nvm_dma_unmap(buffer);
    nvm_aq_destroy(aq_ref);
    nvm_dma_unmap(aq_mem);
    nvm_ctrl_free(ctrl);
    SCITerminate();

    if (status != 0)
    {
        fprintf(stderr, "%d\n", status);
    }

    exit(status);
}
