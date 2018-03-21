#include <nvm_dma.h>
#include <nvm_types.h>
#include <nvm_aq.h>
#include <nvm_admin.h>
#include <nvm_ctrl.h>
#include <nvm_util.h>
#include <sisci_types.h>
#include <sisci_api.h>
#include <getopt.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <limits.h>


/*
 * Command line arguments.
 */
struct cl_args
{
    uint64_t    dev_id;
    uint32_t    dis_adapter;
    uint32_t    segment_id;
};


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


static void parse_args(int argc, char** argv, struct cl_args* args);


int main(int argc, char** argv)
{
    sci_error_t err;
    struct cl_args args;

    parse_args(argc, argv, &args);

    SCIInitialize(0, &err);

    nvm_ctrl_t* ctrl;
    int status = nvm_dis_ctrl_init(&ctrl, args.dev_id, args.dis_adapter);
    if (status != 0)
    {
        fprintf(stderr, "Failed to initialize controller reference: %s\n", strerror(status));
        exit(status);
    }

    nvm_dma_t* window;
    status = nvm_dis_dma_create(&window, ctrl, args.dis_adapter, args.segment_id, 3 * 0x1000);
    if (status != 0)
    {
        nvm_ctrl_free(ctrl);
        fprintf(stderr, "Failed to create local segment: %s\n", strerror(status));
        exit(status);
    }
    memset(window->vaddr, 0, 3 * 0x1000);

    fprintf(stderr, "Resetting controller and setting up admin queues...\n");
    nvm_aq_ref aq;
    status = nvm_aq_create(&aq, ctrl, window);
    if (status != 0)
    {
        nvm_dma_unmap(window);
        nvm_ctrl_free(ctrl);
        fprintf(stderr, "Failed to create admin queues: %s\n", strerror(status));
        exit(status);
    }

    uint16_t n_cqs = 0;
    uint16_t n_sqs = 0;
    status = nvm_admin_get_num_queues(aq, &n_cqs, &n_sqs);

    struct nvm_ctrl_info info;
    status = nvm_admin_ctrl_info(aq, &info, NVM_DMA_OFFSET(window, 2), window->ioaddrs[2]);

    if (status == 0)
    {
        print_ctrl_info(stdout, &info, n_cqs, n_sqs);
    }

    nvm_aq_destroy(aq);
    nvm_dma_unmap(window);
    nvm_ctrl_free(ctrl);
    SCITerminate();
    exit(status);
}


static int parse_u64(const char* str, uint64_t* num, int base)
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


static int parse_u32(const char* str, uint32_t* num, int base)
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


static void give_usage(const char* name)
{
    fprintf(stderr, "Usage: %s --ctrl=<dev id> [--adapter=<adapter>] [--id=<segment id>]\n", name);
}


static void show_help(const char* name)
{
    give_usage(name);
    fprintf(stderr, "\nCreate a manager and run an IDENTIFY CONTROLLER NVM admin command.\n\n"
            "    --ctrl     <fdid>          SmartIO device identifier (fabric device id).\n"
            "    --adapter  <adapter>       DIS adapter number (defaults to 0).\n"
            "    --id       <segment id>    SISCI segment identifier (defaults to 0).\n"
            "    --help                     Show this information.\n\n");
}




static void parse_args(int argc, char** argv, struct cl_args* args)
{
    static struct option opts[] = {
        { "help", no_argument, NULL, 'h' },
        { "ctrl", required_argument, NULL, 'c' },
        { "adapter", required_argument, NULL, 'a' },
        { "id", required_argument, NULL, 'i' },
        { NULL, 0, NULL, 0 }
    };

    int opt;
    int idx;

    bool dev_set = false;
    args->dev_id = 0;
    args->dis_adapter = 0;
    args->segment_id = 0;

    while ((opt = getopt_long(argc, argv, ":hc:a:i:", opts, &idx)) != -1)
    {
        switch (opt)
        {
            case '?': // unknown option
                fprintf(stderr, "Unknown option: `%s'\n", argv[optind - 1]);
                give_usage(argv[0]);
                exit('?');

            case ':': // missing option argument
                fprintf(stderr, "Missing argument for option: `%s'\n", argv[optind - 1]);
                give_usage(argv[0]);
                exit(':');

            case 'c': // device identifier
                dev_set = true;
                if (parse_u64(optarg, &args->dev_id, 16) != 0)
                {
                    give_usage(argv[0]);
                    exit('c');
                }
                break;

            case 'a':
                if (parse_u32(optarg, &args->dis_adapter, 10) != 0)
                {
                    give_usage(argv[0]);
                    exit('a');
                }
                break;

            case 'i':
                if (parse_u32(optarg, &args->segment_id, 0) != 0)
                {
                    give_usage(argv[0]);
                    exit('i');
                }
                break;

            case 'h':
                show_help(argv[0]);
                exit(0);
        }
    }

    if (!dev_set)
    {
        fprintf(stderr, "Device ID is not set!\n");
        give_usage(argv[0]);
        exit('c');
    }
}
