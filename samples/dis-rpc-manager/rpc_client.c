#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_rpc.h>
#include <nvm_dma.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <getopt.h>
#include <string.h>
#include <sisci_api.h>
#include "dis/segment.h"
#include "dis/node.h"
#include "util.h"


/* Program options */
struct cl_args
{
    uint64_t    smartio_dev_id;     // Specify SmartIO device
    uint32_t    ctrl_adapter;       // Controller adapter
    uint32_t    mngr_node_id;       // Manager node identifier
    uint32_t    mngr_adapter;       // Manager adapter
    uint32_t    mngr_intr_no;       // Manager interrupt number
};


static void give_usage(const char* program_name)
{
    fprintf(stderr, 
            "Usage: %s --ctrl <dev id> [--adapter <adapter>] --manager <node id> [--manager-adapter adapter] --intr <intr no>\n",
            program_name);
}


static void parse_opts(int argc, char** argv, struct cl_args* args);


static int show_info(nvm_rpc_t rpc, nvm_ctrl_t ctrl, uint32_t adapter)
{
    struct segment memory_page;
    nvm_dma_t dma;

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

    nvm_ctrl_info_t info;
    status = nvm_rpc_ctrl_info(&info, rpc, ctrl, dma->vaddr, dma->ioaddrs[0]);
    if (status == 0)
    {
        print_ctrl_info(stdout, &info);
    }
    else
    {
        fprintf(stderr, "RPC command request failed: %s\n", strerror(status));
    }

    dma_remove(&dma, &memory_page, adapter);
    segment_remove(&memory_page);
    return status;
}


int main(int argc, char** argv)
{
    nvm_ctrl_t ctrl;
    nvm_rpc_t rpc;

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
    int status = nvm_dis_ctrl_init(&ctrl, args.smartio_dev_id, args.ctrl_adapter);
    if (status != 0)
    {
        fprintf(stderr, "Failed to get controller reference: %s\n", strerror(status));
        exit(2);
    }

    // Get RPC reference
    status = nvm_dis_rpc_bind(&rpc, args.mngr_node_id, args.mngr_intr_no, args.mngr_adapter);
    if (status != 0)
    {
        nvm_ctrl_free(ctrl);
        fprintf(stderr, "Failed to get RPC reference: %s\n", strerror(status));
        exit(2);
    }

    status = show_info(rpc, ctrl, args.ctrl_adapter);

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
            "    Run controller manager RPC client in a DIS cluster.\n\n"
            "    --ctrl             <dev id>    SmartIO device identifier.\n"
            "    --ctrl-adapter     <adapter>   Local adapter to reach device (default is 0).\n"
            "    --manager          <node id>   Remote node manager is running on.\n"
            "    --manager-adapter  <adapter>   Local adapter to reach manager node (default is 0).\n"
            "    --intr             <intr no>   RPC interrupt number.\n"
            "    --help                         Show this information.\n"
            "\n");
}


static void parse_opts(int argc, char** argv, struct cl_args* args)
{
    static struct option opts[] = {
        { "help", no_argument, NULL, 'h' },
        { "ctrl", required_argument, NULL, 'c' },
        { "ctrl-adapter", required_argument, NULL, 'a' },
        { "adapter", required_argument, NULL, 'a' },
        { "manager", required_argument, NULL, 'm' },
        { "mngr", required_argument, NULL, 'm' },
        { "manager-adapter", required_argument, NULL, 'r' },
        { "mngr-adapter", required_argument, NULL, 'r' },
        { "interrupt", required_argument, NULL, 'i' },
        { "intr", required_argument, NULL, 'i' },
        { NULL, 0, NULL, 0 }
    };

    int opt;
    int idx;

    memset(args, 0, sizeof(struct cl_args));
    bool intr_set = false;
    bool node_set = false;

    // Parse arguments
    while ((opt = getopt_long(argc, argv, ":hc:a:m:r:i:", opts, &idx)) != -1)
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

            case 'c': // device identifier
                if (parse_u64(optarg, &args->smartio_dev_id, 0) != 0)
                {
                    fprintf(stderr, "Invalid device id: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('c');
                }
                break;

            case 'a': // device adapter
                if (parse_u32(optarg, &args->ctrl_adapter, 10) != 0)
                {
                    fprintf(stderr, "Invalid adapter number: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('a');
                }
                break;

            case 'm': // remote manager
                if (get_node_info(optarg, &args->mngr_node_id, &args->mngr_adapter) != 0)
                {
                    fprintf(stderr, "Invalid node identifier: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('m');
                }
                node_set = true;
                break;

            case 'i': // interrupt number
                if (parse_u32(optarg, &args->mngr_intr_no, 10) != 0)
                {
                    fprintf(stderr, "Invalid interrupt number: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('i');
                }
                intr_set = true;
                break;

            case 'r': // manager adapter
                if (parse_u32(optarg, &args->mngr_adapter, 0) != 0)
                {
                    fprintf(stderr, "Invalid adapter number: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('r');
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

    if (!intr_set)
    {
        fprintf(stderr, "RPC interrupt number is not set!\n");
        give_usage(argv[0]);
        exit('i');
    }

    if (!node_set)
    {
        fprintf(stderr, "Manager node identifier is not set!\n");
        give_usage(argv[0]);
        exit('m');
    }
}

