#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_manager.h>
#include <nvm_rpc.h>
#include <nvm_dma.h>
#include <nvm_util.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <getopt.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sisci_api.h>
#include "dis/segment.h"
#include "dis/node.h"
#include "util.h"

#define MAX_ADAPTS 4


/* Program options */
struct cl_args
{
    bool        identify_ctrl;      // Indicates if the manager should run an NVM identify controller command
    uint64_t    smartio_dev_id;     // Specify SmartIO device
    uint32_t    ctrl_adapter;       // Controller adapter
    uint32_t    intr_no;            // Specify local data interrupt number
    size_t      n_dis_adapters;     // Number of adapters specified
    uint32_t    dis_adapters[MAX_ADAPTS]; // Indicate which local adapters to bind manager to
};


static bool verbose = false;


/* Indicate whether or not to terminate the manager */
static bool signal_caught = false;


/* Blocking queue to suspend process execution while waiting for signal */
static pthread_cond_t signal_wq = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t signal_lk = PTHREAD_MUTEX_INITIALIZER;


static void give_usage(const char* program_name);
static void show_help(const char* program_name);
static void parse_opts(int argc, char** argv, struct cl_args* args);
static void identify_controller(nvm_ctrl_t ctrl, uint32_t adapter, nvm_manager_t mngr);


/* Signal handler routine */
static void catch_signal()
{
    if (verbose)
    {
        fprintf(stderr, "Signal caught, quiting...\n");
    }

    pthread_mutex_lock(&signal_lk);
    signal_caught = true;
    pthread_cond_signal(&signal_wq);
    pthread_mutex_unlock(&signal_lk);
}


static bool request_accepter(nvm_cmd_t* cmd, uint32_t node_id, uint32_t adapter, uint32_t intr_no)
{
    if (verbose)
    {
        uint16_t cid = *CMD_CID(cmd);
        uint8_t opc = _RB(cmd->dword[0], 7, 0);
        fprintf(stderr, "* Command request from node %u on adapter %u -- cid=0x%04x opc=0x%02x\n", 
                node_id, adapter, cid, opc);
    }

    return true;
}


static int run_service(nvm_ctrl_t ctrl, nvm_dma_t q_wnd, struct cl_args* args)
{
    int status;
    nvm_manager_t mngr;

    status = nvm_manager_register(&mngr, ctrl, q_wnd);
    if (status != 0)
    {
        fprintf(stderr, "Failed to register as manager: %s\n", strerror(status));
        return 2;
    }

    if (args->identify_ctrl)
    {
        identify_controller(ctrl, args->ctrl_adapter, mngr);
    }

    for (size_t i_adapter = 0; i_adapter < args->n_dis_adapters; ++i_adapter)
    {
        // Get local node ID
        uint32_t node_id = 0;
        status = get_local_node_id(args->dis_adapters[i_adapter], &node_id);
        if (status != 0)
        {
            fprintf(stderr, "Unexpected error while getting local node id\n");
        }

        // Enable RPC on adapter
        status = nvm_dis_rpc_enable(mngr, args->dis_adapters[i_adapter], args->intr_no, request_accepter);   
        if (status != 0)
        {
            fprintf(stderr, "Unexpected error: %s\n", strerror(status));
        }

        fprintf(stderr, "Connect RPC client to controller manager on node %u using interrupt %u\n",
                node_id, args->intr_no);
    }

    // Set current thread in background
    pthread_mutex_lock(&signal_lk);
    signal(SIGTERM, (sig_t) catch_signal);
    signal(SIGINT, (sig_t) catch_signal);

    while ( ! signal_caught )
    {
        pthread_cond_wait(&signal_wq, &signal_lk);
    }
    pthread_mutex_unlock(&signal_lk);

    nvm_manager_unregister(mngr);
    return 0;
}


int main(int argc, char** argv)
{
    nvm_ctrl_t ctrl;
    struct segment segment;
    nvm_dma_t q_wnd;

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
    if (verbose)
    {
        fprintf(stderr, "Resetting controller...\n");
    }

    int status = nvm_dis_ctrl_init(&ctrl, args.smartio_dev_id, args.ctrl_adapter);
    if (status != 0)
    {
        fprintf(stderr, "Failed to get controller reference: %s\n", strerror(status));
        exit(2);
    }

    // Create queue memory
    status = segment_create(&segment, random_id(), ctrl->page_size * 2);
    if (status != 0)
    {
        nvm_ctrl_free(ctrl);
        exit(1);
    }

    // Map queue as DMA window
    status = dma_create(&q_wnd, ctrl, &segment, args.ctrl_adapter);
    if (status != 0)
    {
        segment_remove(&segment);
        nvm_ctrl_free(ctrl);
        exit(status);
    }

    // Run manager RPC service
    if (verbose)
    {
        fprintf(stderr, "Starting service...\n");
    }
    status = run_service(ctrl, q_wnd, &args);

    // Destroy queue memory
    dma_remove(&q_wnd, &segment, args.ctrl_adapter);
    segment_remove(&segment);

    // Put controller reference
    nvm_ctrl_free(ctrl);

    SCITerminate();
    
    fprintf(stderr, "Goodbye!\n");
    exit(status);
}


static void parse_opts(int argc, char** argv, struct cl_args* args)
{
    static struct option opts[] = {
        { "help", no_argument, NULL, 'h' },
        { "identify", no_argument, NULL, 1 },
        { "interrupt", required_argument, NULL, 'i' },
        { "intr", required_argument, NULL, 'i' },
        { "enable", required_argument, NULL, 'r' },
        { "ctrl", required_argument, NULL, 'c' },
        { "adapter", required_argument, NULL, 'a' },
        { "verbose", no_argument, NULL, 'v' },
        { NULL, 0, NULL, 0 }
    };

    int opt;
    int idx;

    memset(args, 0, sizeof(struct cl_args));

    // Assign interrupts a random number
    args->intr_no = random_id();

    // Parse arguments
    while ((opt = getopt_long(argc, argv, ":hvc:i:a:", opts, &idx)) != -1)
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

            case 'v': // set verbose
                verbose = true;
                break;

            case 1: // identify controller
                args->identify_ctrl = true;
                break;

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

            case 'i': // interrupt number
                if (parse_u32(optarg, &args->intr_no, 10) != 0)
                {
                    fprintf(stderr, "Invalid interrupt number: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('i');
                }
                break;

            case 'r': // RPC adapter number
                if (args->n_dis_adapters == MAX_ADAPTS)
                {
                    fprintf(stderr, "Maximum adapters reached, ignoring\n");
                    break;
                }

                if (parse_u32(optarg, &args->dis_adapters[args->n_dis_adapters], 0) != 0)
                {
                    fprintf(stderr, "Invalid adapter number: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('r');
                }
                args->n_dis_adapters++;
                break;
        }
    }

    if (args->smartio_dev_id == 0)
    {
        fprintf(stderr, "No controller specified!\n");
        give_usage(argv[0]);
        exit('c');
    }

    if (args->n_dis_adapters == 0)
    {
        args->dis_adapters[args->n_dis_adapters++] = args->ctrl_adapter;
    }
}


static void identify_controller(nvm_ctrl_t ctrl, uint32_t adapter, nvm_manager_t mngr)
{
    int status;
    struct segment segment;
    nvm_dma_t identify_wnd;
    nvm_rpc_t rpc_ref;

    status = segment_create(&segment, random_id(), 0x1000);
    if (status != 0)
    {
        return;
    }

    status = dma_create(&identify_wnd, ctrl, &segment, adapter);
    if (status != 0)
    {
        segment_remove(&segment);
        return;
    }

    status = nvm_rpc_bind_local(&rpc_ref, mngr);
    if (status != 0)
    {
        fprintf(stderr, "Failed to bind RPC client to manager: %s\n", strerror(status));
        dma_remove(&identify_wnd, &segment, adapter);
        segment_remove(&segment);
        return;
    }

    nvm_ctrl_info_t info;
    status = nvm_rpc_ctrl_info(&info, rpc_ref, ctrl, identify_wnd->vaddr, identify_wnd->ioaddrs[0]);
    if (status == 0)
    {
        print_ctrl_info(stdout, &info);
    }

    nvm_rpc_unbind(rpc_ref);
    dma_remove(&identify_wnd, &segment, adapter);
    segment_remove(&segment);
}


static void give_usage(const char* program_name)
{
    fprintf(stderr, 
            "Usage: %s --ctrl <dev id> [--adapter <adapter>] [--intr <intr no>] [--enable <adapter>]...\n", 
            program_name);
}


static void show_help(const char* program_name)
{
    give_usage(program_name);
    fprintf(stderr, 
            "    Run controller manager RPC server in a DIS cluster.\n\n"
            "    --ctrl             <dev id>    SmartIO device identifier.\n"
            "    --adapter          <adapter>   Local adapter to reach device (default is 0).\n"
            "    --intr             <intr no>   RPC interrupt number (defaults to a random number).\n"
            "    --enable           <adapter>   Enable RPC on adapter (defaults to controller adapter).\n"
            "    --identify                     Print controller information.\n"
            "    --verbose                      Print more information.\n"
            "    --help                         Show this information.\n"
            "\n");
}

