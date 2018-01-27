#include "read.h"
#include "args.h"
#include <nvm_types.h>
#include <nvm_admin.h>
#include <nvm_util.h>
#include <nvm_queue.h>
#include <nvm_cmd.h>
#include <nvm_error.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>


static void print_ctrl_info(FILE* fp, const struct nvm_ctrl_info* info)
{
    unsigned char vendor[4];
    memcpy(vendor, &info->pci_vendor, sizeof(vendor));

    char serial[21];
    memset(serial, 0, 21);
    memcpy(serial, info->serial_no, 20);

    char model[41];
    memset(model, 0, 41);
    memcpy(model, info->model_no, 40);

    char revision[9];
    memset(revision, 0, 9);
    memcpy(revision, info->firmware, 8);

    fprintf(fp, "------------- Controller information -------------\n");
    fprintf(fp, "PCI Vendor ID           : %x %x\n", vendor[0], vendor[1]);
    fprintf(fp, "PCI Subsystem Vendor ID : %x %x\n", vendor[2], vendor[3]);
    fprintf(fp, "NVM Express version     : %u.%u.%u\n",
            info->nvme_version >> 16, (info->nvme_version >> 8) & 0xff, info->nvme_version & 0xff);
    fprintf(fp, "Controller page size    : %zu\n", info->page_size);
    fprintf(fp, "Max queue entries       : %u\n", info->max_entries);
    fprintf(fp, "Serial Number           : %s\n", serial);
    fprintf(fp, "Model Number            : %s\n", model);
    fprintf(fp, "Firmware revision       : %s\n", revision);
    fprintf(fp, "Max data transfer size  : %zu\n", info->max_data_size);
    fprintf(fp, "Max outstanding commands: %zu\n", info->max_out_cmds);
    fprintf(fp, "Max number of namespaces: %zu\n", info->max_n_ns);
    fprintf(fp, "--------------------------------------------------\n");
}


int get_disk_info(nvm_aq_ref ref, struct disk_info* info, uint32_t ns_id, void* ptr, uint64_t ioaddr)
{
    int status;
    struct nvm_ctrl_info ctrl_info;
    struct nvm_ns_info ns_info;

    status = nvm_admin_ctrl_info(ref, &ctrl_info, ptr, ioaddr);
    if (!nvm_ok(status))
    {
        fprintf(stderr, "Failed to identify controller: %s\n", nvm_strerror(status));
        return status;
    }

    info->page_size = ctrl_info.page_size;
    info->max_data_size = ctrl_info.max_data_size;

    status = nvm_admin_ns_info(ref, &ns_info, ns_id, ptr, ioaddr);
    if (!nvm_ok(status))
    {
        fprintf(stderr, "Failed identify namespace: %s\n", nvm_strerror(status));
        return status;
    }

    info->ns_id = ns_info.ns_id;
    info->block_size = ns_info.lba_data_size;

    print_ctrl_info(stderr, &ctrl_info);
    return 0;
}



int create_queue_pair(nvm_aq_ref ref, struct queue_pair* qp, nvm_dma_t* cq_mem, nvm_dma_t* sq_mem)
{
    int status;

    status = nvm_admin_set_num_queues(ref, 1, 1);
    if (!nvm_ok(status))
    {
        fprintf(stderr, "Failed to set number of queues: %s\n", nvm_strerror(status));
        return status;
    }

    memset(cq_mem->vaddr, 0, cq_mem->page_size);
    status = nvm_admin_cq_create(ref, &qp->cq, 2, cq_mem->vaddr, cq_mem->ioaddrs[0]);
    if (!nvm_ok(status))
    {
        fprintf(stderr, "Failed to create completion queue: %s\n", nvm_strerror(status));
        return status;
    }

    memset(sq_mem->vaddr, 0, cq_mem->page_size);

    status = nvm_admin_sq_create(ref, &qp->sq, &qp->cq, 2, sq_mem->vaddr, sq_mem->ioaddrs[0]);
    if (!nvm_ok(status))
    {
        fprintf(stderr, "Failed to create submission queue: %s\n", nvm_strerror(status));
        return status;
    }

    qp->sq_mem = sq_mem;
    qp->cq_mem = cq_mem;
    qp->stop = false;
    qp->num_cpls = 0;
    return 0;
}



static void consume_completions(struct queue_pair* qp)
{
    nvm_cpl_t* cpl;

    while (!qp->stop)
    {
        if ((cpl = nvm_cq_dequeue_block(&qp->cq, 100)) == NULL)
        {
            continue;
        }
        nvm_sq_update(&qp->sq);

        if (!NVM_ERR_OK(cpl))
        {
            fprintf(stderr, "%s\n", nvm_strerror(NVM_ERR_STATUS(cpl)));
        }

        nvm_cq_update(&qp->cq);
        qp->num_cpls++;
    }
}



static void dump_memory(const nvm_dma_t* buffer, const struct options* args, size_t size)
{
    if (args->output != NULL)
    {
        FILE* fp = fopen(args->output, "w");
        if (fp == NULL)
        {
            fprintf(stderr, "Failed to open output file: %s\n", strerror(errno));
            return;
        }

        fwrite(buffer->vaddr, 1, size, fp);
        fflush(fp);
        fclose(fp);
        return;
    }

    uint8_t* ptr = (uint8_t*) buffer->vaddr;
    size_t byte = 0;
    while (byte < size)
    {
        fprintf(stdout, "%8lx: ", byte);
        for (size_t n = byte + (args->ascii ? 0x80 : 0x20); byte < n; ++byte)
        {
            uint8_t value = ptr[byte];
            if (args->ascii)
            {
                if ( !(0x20 <= value && value <= 0x7e) )
                {
                    value = ' ';
                }
                fprintf(stdout, "%c", value);
            }
            else
            {
                fprintf(stdout, " %02x", value);
            }
        }
        fprintf(stdout, "\n");
    }
}


int read_and_dump(const struct disk_info* disk, struct queue_pair* qp, const nvm_dma_t* buffer, const struct options* args)
{
    int status;
    pthread_t completer;

    // Start consuming
    status = pthread_create(&completer, NULL, (void *(*)(void*)) consume_completions, qp);
    if (status != 0)
    {
        fprintf(stderr, "Could not start completer thread\n");
        return status;
    }

    // Clear all PRP lists
    memset(NVM_DMA_OFFSET(qp->sq_mem, 1), 0, qp->sq_mem->page_size * (qp->sq_mem->n_ioaddrs - 1));

    // Read blocks
    size_t page = 0;
    size_t num_cmds = 0;
    size_t num_pages = disk->max_data_size / disk->page_size;
    while ((buffer->n_ioaddrs - page) > 0)
    {
        if ((buffer->n_ioaddrs - page) < num_pages)
        {
            num_pages = buffer->n_ioaddrs - page;
        }
    
        nvm_cmd_t* cmd;
        while ((cmd = nvm_sq_enqueue(&qp->sq)) == NULL)
        {
            nvm_sq_submit(&qp->sq);
        }

        uint16_t prp_list = (*NVM_CMD_CID(cmd) % qp->sq.max_entries) + 1;
        size_t num_blocks = NVM_PAGE_TO_BLOCK(disk->page_size, disk->block_size, num_pages);
        size_t start_block = args->offset + NVM_PAGE_TO_BLOCK(disk->page_size, disk->block_size, page);

        nvm_cmd_header(cmd, NVM_IO_READ, disk->ns_id);

        page += nvm_cmd_data(cmd, disk->page_size, num_pages, NVM_DMA_OFFSET(qp->sq_mem, prp_list),
                qp->sq_mem->ioaddrs[prp_list], &buffer->ioaddrs[page]);

        nvm_cmd_rw_blks(cmd, start_block, num_blocks);

        ++num_cmds;
    }

    nvm_sq_submit(&qp->sq);

    // Wait for completions
    while (qp->num_cpls < num_cmds);
    qp->stop = true;
    pthread_join(completer, NULL);

    dump_memory(buffer, args, args->num_blocks * disk->block_size);
    return 0;
}


