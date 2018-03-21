#include "util.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <errno.h>
#include <nvm_types.h>
#include <limits.h>


int parse_u64(const char* str, uint64_t* num, int base)
{
    char* endptr = NULL;
    uint64_t ul = strtoul(str, &endptr, base);

    if (endptr == NULL || *endptr != '\0')
    {
        return EINVAL;
    }

    *num = ul;
    return 0;
}


int parse_u32(const char* str, uint32_t* num, int base)
{
    int status;
    uint64_t ul;

    status = parse_u64(str, &ul, base);

    if (status != 0 || ul > UINT_MAX)
    {
        return EINVAL;
    }

    *num = (uint32_t) ul;
    return status;
}


int parse_u16(const char* str, uint16_t* num, int base)
{
    int status;
    uint64_t ul;

    status = parse_u64(str, &ul, base);

    if (status != 0 || ul > 0xffff)
    {
        return EINVAL;
    }

    *num = (uint16_t) ul;
    return status;
}




uint16_t random_id()
{
    static uint16_t unique_counter = 0;
    struct timeval tv;

    srand(time(NULL));
    
    if (gettimeofday(&tv, NULL) == 0)
    {
        srand(tv.tv_usec);
    }

    return (rand() + unique_counter++) & 0xffff;
}


void print_ctrl_info(FILE* fp, const struct nvm_ctrl_info* info)
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


void print_ns_info(FILE* fp, const struct nvm_ns_info* info)
{
    fprintf(fp, "------------- Namespace  information -------------\n");
    fprintf(fp, "Namespace identifier    : %x\n", info->ns_id);
    fprintf(fp, "Logical block size      : %zu bytes\n", info->lba_data_size);
    fprintf(fp, "Namespace size          : %zu blocks\n", info->size);
    fprintf(fp, "Namespace capacity      : %zu blocks\n", info->capacity);
    fprintf(fp, "--------------------------------------------------\n");
}

