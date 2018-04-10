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
#include "common.h"


/*
 * Command line arguments.
 */
struct cl_args
{
    uint64_t    dev_id;
    uint32_t    dis_adapter;
    uint32_t    segment_id;
    uint32_t    namespace_id;
};


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

    nvm_aq_ref aq;
    aq = reset_ctrl(ctrl, window);
    if (aq == NULL)
    {
        status = 1;
        goto leave;
    }

    status = identify_ctrl(aq, NVM_DMA_OFFSET(window, 2), window->ioaddrs[2]);
    if (status != 0)
    {
        goto leave;
    }

    if (args.namespace_id != 0)
    {
        status = identify_ns(aq, args.namespace_id, NVM_DMA_OFFSET(window, 2), window->ioaddrs[2]);
    }

leave:
    nvm_aq_destroy(aq);
    nvm_dma_unmap(window);
    nvm_ctrl_free(ctrl);
    SCITerminate();

    fprintf(stderr, "Goodbye!\n");
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
            "    --ns       <namespace id>  Show information about NVM namespace.\n"
            "    --adapter  <adapter>       DIS adapter number (defaults to 0).\n"
            "    --segment  <segment id>    SISCI segment identifier (defaults to 0).\n"
            "    --help                     Show this information.\n\n");
}




static void parse_args(int argc, char** argv, struct cl_args* args)
{
    static struct option opts[] = {
        { "help", no_argument, NULL, 'h' },
        { "ctrl", required_argument, NULL, 'c' },
        { "adapter", required_argument, NULL, 'a' },
        { "ns", required_argument, NULL, 'n' },
        { "segment", required_argument, NULL, 's' },
        { NULL, 0, NULL, 0 }
    };

    int opt;
    int idx;

    bool dev_set = false;
    args->dev_id = 0;
    args->dis_adapter = 0;
    args->segment_id = 0;
    args->namespace_id = 0;

    while ((opt = getopt_long(argc, argv, ":hc:a:s:n:", opts, &idx)) != -1)
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

            case 's':
                if (parse_u32(optarg, &args->segment_id, 0) != 0)
                {
                    give_usage(argv[0]);
                    exit('s');
                }
                break;

            case 'n':
                if (parse_u32(optarg, &args->namespace_id, 0) != 0)
                {
                    give_usage(argv[0]);
                    exit('n');
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
