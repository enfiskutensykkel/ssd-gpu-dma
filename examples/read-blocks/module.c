#include "args.h"
#include "read.h"
#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_dma.h>
#include <nvm_aq.h>
#include <nvm_error.h>
#include <nvm_util.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>



static int prepare_and_read(nvm_aq_ref ref, const struct disk_info* disk, const struct options* args)
{
    int status = 0;

    void* buffer_ptr = NULL;
    nvm_dma_t* buffer = NULL;
    void* queue_ptr = NULL;
    nvm_dma_t* sq_mem = NULL;
    nvm_dma_t* cq_mem = NULL;
    size_t n_prp_lists = disk->page_size / sizeof(nvm_cmd_t);
    struct queue_pair queues;

    const nvm_ctrl_t* ctrl = nvm_ctrl_from_aq_ref(ref);

    status = posix_memalign(&buffer_ptr, disk->page_size, NVM_CTRL_ALIGN(ctrl, args->num_blocks * disk->block_size));
    if (status != 0)
    {
        fprintf(stderr, "Failed to allocate memory buffer: %s\n", strerror(status));
        goto leave;
    }

    status = posix_memalign(&queue_ptr, disk->page_size, disk->page_size * (n_prp_lists + 2));
    if (status != 0)
    {
        fprintf(stderr, "Failed to allocate queue memory: %s\n", strerror(status));
        goto leave;
    }

    status = nvm_dma_map_host(&sq_mem, ctrl, NVM_PTR_OFFSET(queue_ptr, disk->page_size, 1), disk->page_size * (n_prp_lists + 1));
    if (!nvm_ok(status))
    {
        fprintf(stderr, "Failed to map memory for controller: %s\n", nvm_strerror(status));
        goto leave;
    }

    status = nvm_dma_map_host(&cq_mem, ctrl, queue_ptr, disk->page_size);
    if (!nvm_ok(status))
    {
        fprintf(stderr, "Failed to map memory for controller: %s\n", nvm_strerror(status));
        goto leave;
    }

    status = nvm_dma_map_host(&buffer, ctrl, buffer_ptr, args->num_blocks * disk->block_size);
    if (!nvm_ok(status))
    {
        fprintf(stderr, "Failed to map memory for controller: %s\n", nvm_strerror(status));
        goto leave;
    }

    status = create_queue_pair(ref, &queues, cq_mem, sq_mem);
    if (status != 0)
    {
        goto leave;
    }

    if (args->input != NULL)
    {
        status = write_blocks(disk, &queues, buffer, args);
        if (status != 0)
        {
            goto leave;
        }
    }

    status = read_and_dump(disk, &queues, buffer, args);

leave:
    nvm_dma_unmap(buffer);
    nvm_dma_unmap(sq_mem);
    nvm_dma_unmap(cq_mem);
    free(buffer_ptr);
    free(queue_ptr);
    return status;
}



int main(int argc, char** argv)
{
    int status;
    int fd;

    struct disk_info disk;

    nvm_ctrl_t* ctrl = NULL;
    void* aq_ptr = NULL;
    nvm_dma_t* aq_mem = NULL;
    nvm_aq_ref aq_ref = NULL;

    struct options args;

    // Parse arguments from command line
    parse_options(argc, argv, &args);

    // Get controller reference
    fd = open(args.controller_path, O_RDWR | O_NONBLOCK);
    if (fd < 0)
    {
        fprintf(stderr, "Failed to open file descriptor: %s\n", strerror(errno));
        exit(1);
    }

    status = nvm_ctrl_init(&ctrl, fd);
    if (!nvm_ok(status))
    {
        fprintf(stderr, "Failed to initialize controller reference: %s\n", nvm_strerror(status));
        goto leave;
    }

    // Create admin queue pair + page for identify commands
    status = posix_memalign(&aq_ptr, ctrl->page_size, ctrl->page_size * 3);
    if (status != 0)
    {
        fprintf(stderr, "Failed to allocate queue memory: %s\n", strerror(status));
        goto leave;
    }

    status = nvm_dma_map_host(&aq_mem, ctrl, aq_ptr, ctrl->page_size * 3);
    if (!nvm_ok(status))
    {
        fprintf(stderr, "Failed to map memory for controller: %s\n", nvm_strerror(status));
        goto leave;
    }

    // Reset controller and set admin queues
    status = nvm_aq_create(&aq_ref, ctrl, aq_mem);
    if (!nvm_ok(status))
    {
        fprintf(stderr, "Failed to reset controller: %s\n", nvm_strerror(status));
        goto leave;
    }

    // Identify controller and namespace
    status = get_disk_info(aq_ref, &disk, args.namespace_id, NVM_DMA_OFFSET(aq_mem, 2), aq_mem->ioaddrs[2], args.identify);
    if (status != 0)
    {
        goto leave;
    }

    status = prepare_and_read(aq_ref, &disk, &args);

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

    nvm_aq_destroy(aq_ref);
    nvm_dma_unmap(aq_mem);
    free(aq_ptr);
    nvm_ctrl_free(ctrl);
    close(fd);
    exit(status);
}
