#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_aq.h>
#include <nvm_admin.h>
#include <nvm_dma.h>
#include <nvm_util.h>
#include <nvm_error.h>
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
#include "segment.h"
#include "util.h"

#define MAX_ADAPTS NVM_DIS_RPC_MAX_ADAPTER 


/* Program options */
struct cl_args
{
    bool        identify_ctrl;      // Indicates if the manager should run an NVM identify controller command
    uint64_t    smartio_dev_id;     // Specify SmartIO device
    uint32_t    ctrl_adapter;       // Controller adapter
    uint16_t    n_sqs;              // Number of SQs to reserve
    uint16_t    n_cqs;              // Number of CQs to reserve
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
static void identify_controller(const nvm_ctrl_t* ctrl, uint32_t adapter, nvm_aq_ref rpc);


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


static bool request_accepter(nvm_cmd_t* cmd, uint32_t adapter, uint32_t node_id)
{
    if (verbose)
    {
        fprintf(stderr, "* Command request received on adapter %u from node %u\n",
                adapter, node_id);
    }

    return true;
}


static int run_service(const nvm_ctrl_t* ctrl, const nvm_dma_t* q_wnd, const struct cl_args* args)
{
    int status;
    nvm_aq_ref rpc;

    status = nvm_aq_create(&rpc, ctrl, q_wnd);
    if (status != 0)
    {
        fprintf(stderr, "Failed to register as manager: %s\n", strerror(status));
        return 2;
    }

    if (args->identify_ctrl)
    {
        identify_controller(ctrl, args->ctrl_adapter, rpc);
    }

    if (args->n_sqs > 0 && args->n_cqs > 0)
    {
        uint16_t n_sqs = args->n_sqs;
        uint16_t n_cqs = args->n_cqs;
        status = nvm_admin_request_num_queues(rpc, &n_cqs, &n_sqs);
        if (!nvm_ok(status))
        {
            fprintf(stderr, "Failed to run admin command for reserving IO queues: %s\n", nvm_strerror(status));
            nvm_aq_destroy(rpc);
            return 3;
        }

        if (n_sqs < args->n_sqs || n_cqs < args->n_cqs)
        {
            fprintf(stderr, "Attempted to reserve %u CQs and %u SQs, got %u CQs and %u SQs\n",
                    args->n_cqs, args->n_sqs, n_cqs, n_sqs);
            nvm_aq_destroy(rpc);
            return 3;
        }

        if (verbose)
        {
            fprintf(stderr, "Reserved %u CQs and %u SQs\n", n_cqs, n_sqs);
        }
    }

    for (size_t i_adapter = 0; i_adapter < args->n_dis_adapters; ++i_adapter)
    {
        // Enable RPC on adapter
        status = nvm_dis_rpc_enable(rpc, args->dis_adapters[i_adapter], request_accepter);   
        if (status != 0)
        {
            fprintf(stderr, "Unexpected error: %s\n", strerror(status));
        }
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

    nvm_aq_destroy(rpc);
    return 0;
}


int main(int argc, char** argv)
{
    nvm_ctrl_t* ctrl;
    struct segment segment;
    nvm_dma_t* q_wnd;

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
    dma_remove(q_wnd, &segment, args.ctrl_adapter);
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
        { "sqs", required_argument, NULL, 's' },
        { "cqs", required_argument, NULL, 'c' },
        { "enable", required_argument, NULL, 'r' },
        { "ctrl", required_argument, NULL, 'd' },
        { "adapter", required_argument, NULL, 'a' },
        { "verbose", no_argument, NULL, 'v' },
        { NULL, 0, NULL, 0 }
    };

    int opt;
    int idx;

    memset(args, 0, sizeof(struct cl_args));

    // Parse arguments
    while ((opt = getopt_long(argc, argv, ":hvd:c:a:s:", opts, &idx)) != -1)
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

            case 'd': // device identifier
                if (parse_u64(optarg, &args->smartio_dev_id, 16) != 0)
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

            case 's': // Set number of SQs to reserve
                if (parse_u16(optarg, &args->n_sqs, 0) != 0)
                {
                    fprintf(stderr, "Invalid number of IO submission queues: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('q');
                }
                break;

            case 'c': // Set number of CQs to reserve
                if (parse_u16(optarg, &args->n_cqs, 0) != 0)
                {
                    fprintf(stderr, "Invalid number of IO completion queues: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('q');
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

    if (args->n_dis_adapters == 0)
    {
        args->dis_adapters[args->n_dis_adapters++] = args->ctrl_adapter;
    }

    if ((args->n_cqs > 0 && args->n_sqs == 0) || (args->n_sqs > 0 && args->n_cqs == 0))
    {
        fprintf(stderr, "Must specify either both number of submission and completion queues, or none at all\n");
        give_usage(argv[0]);
        exit('q');
    }
}


static void identify_controller(const nvm_ctrl_t* ctrl, uint32_t adapter, nvm_aq_ref rpc)
{
    int status;
    struct segment segment;
    nvm_dma_t* identify_wnd;
    struct nvm_ctrl_info info;

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

    status = nvm_admin_ctrl_info(rpc, &info, identify_wnd->vaddr, identify_wnd->ioaddrs[0]);
    if (status == 0)
    {
        print_ctrl_info(stdout, &info);
    }

    dma_remove(identify_wnd, &segment, adapter);
    segment_remove(&segment);
}


static void give_usage(const char* program_name)
{
    fprintf(stderr, 
            "Usage: %s --ctrl <fdid> [--adapter <adapter>] [--enable <adapter>]...\n", 
            program_name);
}


static void show_help(const char* program_name)
{
    give_usage(program_name);
    fprintf(stderr, 
            "\nRun controller manager RPC server in a DIS cluster.\n\n"
            "    --ctrl             <fdid>      SmartIO device identifier.\n"
            "    --adapter          <adapter>   Local adapter to reach device (default is 0).\n"
            "    --enable           <adapter>   Enable RPC on adapter (defaults to controller adapter).\n"
            "    --cqs              <number>    Reserve number of completion queues (default is not to reserve).\n"
            "    --sqs              <number>    Reserver number of submission queues (default is not to reserve).\n"
            "    --identify                     Print controller information.\n"
            "    --verbose                      Print more information.\n"
            "    --help                         Show this information.\n"
            "\n");
}

