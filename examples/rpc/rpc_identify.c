#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_rpc.h>
#include <nvm_dma.h>
#include <nvm_admin.h>
#include <nvm_util.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <getopt.h>
#include <string.h>
#include <sisci_api.h>
#include "segment.h"
#include "util.h"


/* Program options */
struct cl_args
{
    uint64_t    smartio_dev_id;     // Specify SmartIO device
    uint32_t    nvm_namespace;      // NVM namespace
    uint32_t    adapter;            // Controller adapter
};


static void give_usage(const char* program_name)
{
    fprintf(stderr, 
            "Usage: %s --ctrl <fdid> [--adapter <adapter>] [--namespace <ns id>]\n",
            program_name);
}


static void parse_opts(int argc, char** argv, struct cl_args* args);


static int show_ctrl_info(nvm_aq_ref rpc, const nvm_ctrl_t* ctrl, uint32_t adapter)
{
    struct segment memory_page;
    nvm_dma_t* dma;

    int status = segment_create(&memory_page, random_id(), 0x1000);
    if (status != 0)
    {
        fprintf(stderr, "Failed to create memory segment\n");
        return 1;
    }

    status = dma_create(&dma, ctrl, &memory_page, adapter);
    if (status != 0)
    {
        segment_remove(&memory_page);
        fprintf(stderr, "Failed to create DMA window\n");
        return 2;
    }

    struct nvm_ctrl_info info;
    status = nvm_admin_ctrl_info(rpc, &info, dma->vaddr, dma->ioaddrs[0]);
    if (status == 0)
    {
        print_ctrl_info(stdout, &info);
    }
    else
    {
        fprintf(stderr, "RPC command request failed: %s\n", strerror(status));
    }

    dma_remove(dma, &memory_page, adapter);
    segment_remove(&memory_page);
    return status;
}




static int show_ns_info(nvm_aq_ref rpc, uint32_t ns_id, uint32_t adapter)
{
    struct segment memory_page;
    nvm_dma_t* dma;

    int status = segment_create(&memory_page, random_id(), 0x1000);
    if (status != 0)
    {
        fprintf(stderr, "Failed to create memory segment\n");
        return 1;
    }

    status = dma_create(&dma, nvm_ctrl_from_aq_ref(rpc), &memory_page, adapter);
    if (status != 0)
    {
        segment_remove(&memory_page);
        fprintf(stderr, "Failed to create DMA window\n");
        return 2;
    }

    struct nvm_ns_info info;
    status = nvm_admin_ns_info(rpc, &info, ns_id, dma->vaddr, dma->ioaddrs[0]);
    if (status == 0)
    {
        print_ns_info(stdout, &info);
    }
    else
    {
        fprintf(stderr, "RPC command request failed: %s\n", strerror(status));
    }

    dma_remove(dma, &memory_page, adapter);
    segment_remove(&memory_page);
    return status;
}



int main(int argc, char** argv)
{
    nvm_ctrl_t* ctrl;
    nvm_aq_ref rpc;

    struct cl_args args;
    parse_opts(argc, argv, &args);

    sci_error_t err;
    SCIInitialize(0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to initialize SISCI: %s\n", SCIGetErrorString(err));
        exit(1);
    }

    // Get controller reference
    int status = nvm_dis_ctrl_init(&ctrl, args.smartio_dev_id, args.adapter);
    if (status != 0)
    {
        fprintf(stderr, "Failed to get controller reference: %s\n", strerror(status));
        exit(2);
    }

    // Get RPC reference
    status = nvm_dis_rpc_bind(&rpc, ctrl, args.adapter);
    if (status != 0)
    {
        nvm_ctrl_free(ctrl);
        fprintf(stderr, "Failed to get RPC reference: %s\n", strerror(status));
        exit(2);
    }

    status = show_ctrl_info(rpc, ctrl, args.adapter);
    if (status != 0)
    {
        goto leave;
    }

    if (args.nvm_namespace > 0)
    {
        status = show_ns_info(rpc, args.nvm_namespace, args.adapter);
    }

leave:
    // Free resources and quit
    nvm_rpc_unbind(rpc);
    nvm_ctrl_free(ctrl);
    SCITerminate();
    exit(status);
}


static void show_help(const char* program_name)
{
    give_usage(program_name);
    fprintf(stderr, 
            "\nIdentify controller using a remote controller manager in a DIS cluster.\n\n"
            "    --ctrl             <fdid>      SmartIO device identifier.\n"
            "    --namespace        <ns id>     Show information about NVM namespace.\n"
            "    --adapter          <adapter>   Local adapter to reach device (default is 0).\n"
            "    --help                         Show this information.\n"
            "\n");
}


static void parse_opts(int argc, char** argv, struct cl_args* args)
{
    static struct option opts[] = {
        { "help", no_argument, NULL, 'h' },
        { "ctrl", required_argument, NULL, 'd' },
        { "namespace", required_argument, NULL, 'n' },
        { "ns", required_argument, NULL, 'n' },
        { "adapter", required_argument, NULL, 'a' },
        { NULL, 0, NULL, 0 }
    };

    int opt;
    int idx;

    memset(args, 0, sizeof(struct cl_args));

    // Parse arguments
    while ((opt = getopt_long(argc, argv, ":hd:a:n:", opts, &idx)) != -1)
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

            case 'h': // show help
                show_help(argv[0]);
                exit(0);

            case 'd': // device identifier
                if (parse_u64(optarg, &args->smartio_dev_id, 16) != 0 || args->smartio_dev_id == 0)
                {
                    fprintf(stderr, "Invalid device id: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('c');
                }
                break;

            case 'n': // namespace identifier
                if (parse_u32(optarg, &args->nvm_namespace, 0) != 0 || args->nvm_namespace == 0)
                {
                    fprintf(stderr, "Invalid namespace id: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('n');
                }
                break;

            case 'a': // device adapter
                if (parse_u32(optarg, &args->adapter, 10) != 0)
                {
                    fprintf(stderr, "Invalid adapter number: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('a');
                }
                break;
        }
    }

    if (args->smartio_dev_id == 0)
    {
        fprintf(stderr, "No controller specified!\n");
        give_usage(argv[0]);
        exit('c');
    }
}

