#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_manager.h>
#include <nvm_rpc.h>
#include <nvm_dma.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <getopt.h>
#include <stdlib.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <sisci_types.h>
#include <sisci_api.h>
#include <sisci_error.h>


/* Program options */
struct cl_args
{
    bool                identify_ctrl;      // Indicates if the manager should run an NVM identify controller command
    uint64_t            smartio_dev_id;     // Specify SmartIO device
    uint32_t            ctrl_adapter;       // Controller adapter
    uint32_t            intr_no;            // Specify local data interrupt number
    size_t              n_dis_adapters;     // Number of adapters specified
    uint32_t            dis_adapters[256];  // Indicate which local adapters to bind manager to
};


/* Memory descriptor */
struct segment
{
    sci_desc_t          sd;                 // SISCI virtual device
    sci_local_segment_t segment;            // Local segment descriptor
    sci_map_t           map;                // Mapped memory descriptor
    size_t              size;               // Size of segment
    void*               vaddr;              // Pointer to mapped memory
};


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
    fprintf(stderr, "Signal caught, quiting...\n");

    pthread_mutex_lock(&signal_lk);
    signal_caught = true;
    pthread_cond_signal(&signal_wq);
    pthread_mutex_unlock(&signal_lk);
}


/* Generate a random identifier */
static uint32_t random_id()
{
    srand(time(NULL)); // TODO use usecs instead
    return rand() % 0x10000;
}


/* Create queue memory */
static int init_segment(struct segment* seg, uint32_t segment_id, size_t size)
{
    sci_error_t err;

    SCIOpen(&seg->sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to open descriptor: %s\n", SCIGetErrorString(err));
        return EIO;
    }

    SCICreateSegment(seg->sd, &seg->segment, segment_id, size, NULL, NULL, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to create segment: %s\n", SCIGetErrorString(err));
        SCIClose(seg->sd, 0, &err);
        return ENOSPC;
    }

    seg->vaddr = SCIMapLocalSegment(seg->segment, &seg->map, 0, size, NULL, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to map segment: %s\n", SCIGetErrorString(err));
        SCIRemoveSegment(seg->segment, 0, &err);
        SCIClose(seg->sd, 0, &err);
        return EIO;
    }

    seg->size = size;

    return 0;
}


static int init_window(nvm_dma_t* window, nvm_ctrl_t ctrl, struct segment* segment, uint32_t ctrl_adapter)
{
    sci_error_t err;

    SCIPrepareSegment(segment->segment, ctrl_adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to prepare segment on adapter: %s\n", SCIGetErrorString(err));
        return 1;
    }

    SCISetSegmentAvailable(segment->segment, ctrl_adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to set segment available: %s\n", SCIGetErrorString(err));
        return 1;
    }

    int status = nvm_dis_dma_window_init(window, ctrl, ctrl_adapter, segment->segment, segment->vaddr, segment->size);
    if (status != 0)
    {
        fprintf(stderr, "Failed to create DMA window: %s\n", strerror(status));

        do
        {
            SCISetSegmentUnavailable(segment->segment, ctrl_adapter, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

        return 2;
    }

    return 0;
}


static void free_window(nvm_dma_t* window, struct segment* segment, uint32_t adapter)
{
    sci_error_t err;

    nvm_dma_window_free(*window);

    do
    {
        SCISetSegmentUnavailable(segment->segment, adapter, 0, &err);
    }
    while (err == SCI_ERR_BUSY);
}


/* Destroy queue memory */
static void free_segment(struct segment* seg)
{
    sci_error_t err;

    seg->vaddr = NULL;

    do
    {
        SCIUnmapSegment(seg->map, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    do
    {
        SCIRemoveSegment(seg->segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    SCIClose(seg->sd, 0, &err);
}


static bool request_accepter(uint32_t node_id, uint32_t adapter, uint32_t intr_no, nvm_cmd_t* cmd)
{
    fprintf(stdout, "Got command request from node %u\n", node_id);
    return true;
}


static int run_service(nvm_ctrl_t ctrl, nvm_dma_t q_wnd, struct cl_args* args)
{
    int status;
    sci_error_t err;
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
        SCIGetLocalNodeId(args->dis_adapters[i_adapter], &node_id, 0, &err);
        if (err != SCI_ERR_OK)
        {
            fprintf(stderr, "Unexpected error while getting local node id: %s\n", SCIGetErrorString(err));
        }

        // Enable RPC on adapter
        status = nvm_dis_rpc_enable(mngr, args->dis_adapters[i_adapter], args->intr_no, request_accepter);   
        if (status != 0)
        {
            fprintf(stderr, "Unexpected error: %s\n", strerror(status));
        }

        fprintf(stderr, "Connect to DIS NVMe manager on node %u using interrupt %u\n",
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

    if (args.smartio_dev_id == 0)
    {
        fprintf(stderr, "No controller specified!\n");
        give_usage(argv[0]);
        exit('c');
    }

    if (args.n_dis_adapters == 0)
    {
        args.dis_adapters[args.n_dis_adapters++] = 0;
    }

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

    // Create queue memory
    status = init_segment(&segment, random_id(), ctrl->page_size * 2);
    if (status != 0)
    {
        nvm_ctrl_free(ctrl);
        exit(1);
    }

    // Map queue as DMA window
    status = init_window(&q_wnd, ctrl, &segment, args.ctrl_adapter);
    if (status != 0)
    {
        free_segment(&segment);
        nvm_ctrl_free(ctrl);
        exit(status);
    }

    // Run manager RPC service
    status = run_service(ctrl, q_wnd, &args);

    // Destroy queue memory
    free_window(&q_wnd, &segment, args.ctrl_adapter);
    free_segment(&segment);

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
        { "identify", no_argument, NULL, 'v' },
        { "interrupt", required_argument, NULL, 'i' },
        { "adapter", required_argument, NULL, 'a' },
        { "ctrl", required_argument, NULL, 'c' },
        { NULL, 0, NULL, 0 }
    };

    int opt;
    int idx;
    char* endptr;

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

            case 'v': // identify controller
                args->identify_ctrl = true;
                break;

            case 'c': // device identifier
                endptr = NULL;
                args->smartio_dev_id = strtoul(optarg, &endptr, 0);

                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid device id: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('c');
                }
                break;

            case 'i': // interrupt number
                endptr = NULL;
                args->intr_no = strtoul(optarg, &endptr, 0);

                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid interrupt number: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('i');
                }
                break;

            case 'a': // adapter number
                endptr = NULL;
                args->dis_adapters[args->n_dis_adapters++] = strtoul(optarg, &endptr, 0);

                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid adapter number: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('a');
                }
                break;
        }
    }
}


static void print_controller_info(nvm_ctrl_info_t* info)
{
    unsigned char vendor[4];
    memcpy(vendor, &info->pci_vendor, sizeof(vendor));

    char serial[21];
    memset(serial, 0, 21);
    memcpy(serial, info->serial_no, 20);

    char model[41];
    memset(model, 0, 41);
    memcpy(model, info->model_no, 40);

    fprintf(stdout, "------------- Controller information -------------\n");
    fprintf(stdout, "PCI Vendor ID           : %x %x\n", vendor[0], vendor[1]);
    fprintf(stdout, "PCI Subsystem Vendor ID : %x %x\n", vendor[2], vendor[3]);
    fprintf(stdout, "NVM Express version     : %u.%u.%u\n",
            info->nvme_version >> 16, (info->nvme_version >> 8) & 0xff, info->nvme_version & 0xff);
    fprintf(stdout, "Controller page size    : %zu\n", info->page_size);
    fprintf(stdout, "Serial Number           : %s\n", serial);
    fprintf(stdout, "Model Number            : %s\n", model);
    fprintf(stdout, "Maximum number of CQs   : %zu\n", info->max_cqs);
    fprintf(stdout, "Maximum number of SQs   : %zu\n", info->max_sqs);
    fprintf(stdout, "Number of namespaces    : %zu\n", info->n_ns);
    fprintf(stdout, "--------------------------------------------------\n");
}


static void identify_controller(nvm_ctrl_t ctrl, uint32_t adapter, nvm_manager_t mngr)
{
    int status;
    struct segment segment;
    nvm_dma_t identify_wnd;
    nvm_rpc_t rpc_ref;
    
    status = init_segment(&segment, random_id(), 0x1000);
    if (status != 0)
    {
        return;
    }

    status = init_window(&identify_wnd, ctrl, &segment, adapter);
    if (status != 0)
    {
        free_segment(&segment);
        return;
    }

    status = nvm_rpc_bind_local(&rpc_ref, mngr);
    if (status != 0)
    {
        fprintf(stderr, "Failed to bind RPC client to manager: %s\n", strerror(status));
        free_window(&identify_wnd, &segment, adapter);
        free_segment(&segment);
        return;
    }

    nvm_ctrl_info_t info;
    status = nvm_rpc_identify(rpc_ref, ctrl, identify_wnd, &info);
    if (status == 0)
    {
        print_controller_info(&info);
    }

    nvm_rpc_unbind(rpc_ref);
    free_window(&identify_wnd, &segment, adapter);
    free_segment(&segment);
}


static void give_usage(const char* program_name)
{
    fprintf(stderr, 
            "Usage: %s --ctrl=<SmartIO device> [--identify] [--intr=<interrupt no>] [--adapter=<DIS adapter>...]\n", 
            program_name);
}


static void show_help(const char* program_name)
{
    give_usage(program_name);
    fprintf(stderr, "    Run NVMe queue manager service in a DIS cluster.\n\n"
            "    --ctrl=<SmartIO device>    DIS SmartIO device ID.\n"
            "    --identify                 Print controller information.\n"
            "    --intr=<interrupt no>      Local DIS data interrupt number.\n"
            "    --adapter<DIS adapter>     Local DIS adapter to bind manager to (default 0).\n"
            "\n");
}

