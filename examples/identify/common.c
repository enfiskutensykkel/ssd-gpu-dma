#include <nvm_types.h>
#include <nvm_aq.h>
#include <nvm_admin.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "common.h"



/*
 * Print controller information.
 */
static void print_ctrl_info(FILE* fp, const struct nvm_ctrl_info* info, uint16_t n_cqs, uint16_t n_sqs)
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
    fprintf(fp, "Current number of CQs   : %u\n", n_cqs);
    fprintf(fp, "Current number of SQs   : %u\n", n_sqs);
    fprintf(fp, "--------------------------------------------------\n");
}


/*
 * Print namespace information.
 */
static void print_ns_info(FILE* fp, const struct nvm_ns_info* info)
{
    fprintf(fp, "------------- Namespace  information -------------\n");
    fprintf(fp, "Namespace identifier    : %x\n", info->ns_id);
    fprintf(fp, "Logical block size      : %zu bytes\n", info->lba_data_size);
    fprintf(fp, "Namespace size          : %zu blocks\n", info->size);
    fprintf(fp, "Namespace capacity      : %zu blocks\n", info->capacity);
    fprintf(fp, "--------------------------------------------------\n");
}



nvm_aq_ref reset_ctrl(const nvm_ctrl_t* ctrl, const nvm_dma_t* dma_window)
{
    int status;
    nvm_aq_ref admin;

    if (dma_window->n_ioaddrs < 2)
    {
        return NULL;
    }
    memset(dma_window->vaddr, 0, dma_window->page_size * 2);

    fprintf(stderr, "Resetting controller and setting up admin queues...\n");
    status = nvm_aq_create(&admin, ctrl, dma_window);
    if (status != 0)
    {
        fprintf(stderr, "Failed to reset controller: %s\n", strerror(status));
        return NULL;
    }

    return admin;
}



int identify_ctrl(nvm_aq_ref admin, void* ptr, uint64_t ioaddr)
{
    int status;
    uint16_t n_cqs = 0;
    uint16_t n_sqs = 0;
    struct nvm_ctrl_info info;

    status = nvm_admin_get_num_queues(admin, &n_cqs, &n_sqs);
    if (status != 0)
    {
        fprintf(stderr, "Failed to get number of queues\n");
        return status;
    }

    status = nvm_admin_ctrl_info(admin, &info, ptr, ioaddr);
    if (status != 0)
    {
        fprintf(stderr, "Failed to identify controller: %s\n", strerror(status));
        return status;
    }

    print_ctrl_info(stdout, &info, n_cqs, n_sqs);
    return 0;
}



int identify_ns(nvm_aq_ref admin, uint32_t nvm_namespace, void* ptr, uint64_t ioaddr)
{
    int status;
    struct nvm_ns_info info;

    status = nvm_admin_ns_info(admin, &info, nvm_namespace, ptr, ioaddr);
    if (status != 0)
    {
        fprintf(stderr, "Failed to identify namespace: %s\n", strerror(status));
        return status;
    }

    print_ns_info(stdout, &info);
    return 0;
}

