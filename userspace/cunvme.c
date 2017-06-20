#include "nvm/types.h"
#include "nvm/ctrl.h"
#include "nvm/command.h"
#include "nvm/queue.h"
#include "message.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <getopt.h>
#include <sisci_types.h>
#include <sisci_api.h>
#include <signal.h>
#include <pthread.h>


extern int workload(sci_device_t dev, uint32_t node_id, uint32_t intno, uint32_t ns_id, void* io_mem, size_t io_size);


static volatile int run_master = 1;

//static pthread_cond_t run_master_cv = PTHREAD_COND_INITIALIZER;


static struct option opts[] = {
    { "help", no_argument, NULL, 'h' },
    { "identify", no_argument, NULL, 'c' },
    { "device", required_argument, NULL, 'd' },
    { "master", required_argument, NULL, 'r' },
    { "interrupt", required_argument, NULL, 'i' },
    { NULL, 0, NULL, 0 }
};


static void give_usage(const char* program_name)
{
    fprintf(stderr, "Usage: %s --device=<dev id> [--master=<node id>] [options...]\n", program_name);
}


static void show_help(const char* program_name)
{
    give_usage(program_name);
}


static void shutdown_master()
{
    run_master = 0;
}


static sci_callback_action_t handle_command(
        nvm_ctrl_t* ctrl,
        sci_local_data_interrupt_t intr, 
        struct command* command,
        unsigned length,
        sci_error_t status)
{
    if (length != sizeof(struct command))
    {
        fprintf(stderr, "Received garbage from remote node\n");
        return SCI_CALLBACK_CONTINUE;
    }

    uint32_t node_id = command->dword[14];
    uint32_t intno = command->dword[15];

    // Enqueue command 
    command->dword[14] = command->dword[15] = 0;
    struct command* in_queue = sq_enqueue(&ctrl->admin_sq);
    if (in_queue == NULL)
    {
        fprintf(stderr, "Admin queue was full, discarding command from node %u\n", node_id);
        send_completion(node_id, intno, NULL, 500);
        return SCI_CALLBACK_CONTINUE;
    }

    // Submit command
    memcpy(in_queue, command, sizeof(struct command));
    sq_submit(&ctrl->admin_sq);

    // Wait for completion
    struct completion* cpl = cq_dequeue_block(&ctrl->admin_cq, ctrl->timeout);
    if (cpl == NULL)
    {
        fprintf(stderr, "Timed out while waiting for completion for command from node %u\n", node_id);
        send_completion(node_id, intno, NULL, 500);
        return SCI_CALLBACK_CONTINUE;
    }
    sq_update(&ctrl->admin_sq, cpl);

    // Send completion back
    send_completion(node_id, intno, cpl, 500);
    cq_update(&ctrl->admin_cq);

    return SCI_CALLBACK_CONTINUE;
}


static void print_controller_info(nvm_ctrl_t* controller)
{
    unsigned char* data = controller->identify.virt_addr;

    char serial[21];
    memset(serial, 0, 21);
    memcpy(serial, data + 4, 20);

    char model[41];
    memset(model, 0, 41);
    memcpy(model, data + 24, 40);

    fprintf(stdout, "PCI Vendor ID           : %x %x\n", data[0], data[1]);
    fprintf(stdout, "PCI Subsystem Vendor ID : %x %x\n", data[2], data[3]);
    fprintf(stdout, "Serial Number           : %s\n", serial);
    fprintf(stdout, "Model Number            : %s\n", model);
    fprintf(stdout, "Number of namespaces    : %u\n", controller->n_ns);
    fprintf(stdout, "Maximum number of queues: %u\n", controller->max_queues);
    
    fprintf(stdout, "dstrd: %u\n", controller->dstrd);
}


int main(int argc, char** argv)
{
    int opt;
    int idx;
    sci_error_t err;
    char* endptr;

    int identify = 0;
    uint32_t node_id = 0;
    uint64_t device_id = 0;
    uint32_t intno = 0;

    while ((opt = getopt_long(argc, argv, ":hci:d:r:", opts, &idx)) != -1)
    {
        switch (opt)
        {
            case '?': // unknown option
                fprintf(stderr, "Unknown option: `%s'\n", argv[optind - 1]);
                give_usage(argv[0]);
                return '?';

            case ':': // missing option argument
                fprintf(stderr, "Missing argument for option `%s'\n", argv[optind - 1]);
                give_usage(argv[0]);
                return ':';

            case 'h': // show help
                show_help(argv[0]);
                return 0;

            case 'c': // identify controller
                identify = 1;
                break;

            case 'r': // connect to remote host
                run_master = 0;
                endptr = NULL;
                node_id = strtoul(optarg, &endptr, 0);

                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Not a valid node id: %s\n", optarg);
                    give_usage(argv[0]);
                    return 'r';
                }
                break;

            case 'i': // specify interrupt number
                endptr = NULL;
                intno = strtoul(optarg, &endptr, 0);

                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Not a valid interrupt number: %s\n", optarg);
                    give_usage(argv[0]);
                    return 'i';
                }
                break;

            case 'd': // specify device
                endptr = NULL;
                device_id = strtoul(optarg, &endptr, 0);

                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Not a valid device id: %s\n", optarg);
                    give_usage(argv[0]);
                    return 'd';
                }
                break;
        }
    }

    SCIInitialize(0, &err);

    sci_desc_t sd;
    SCIOpen(&sd, 0, &err);

    // Connect to device and get valid device handle to NVMe controller
    sci_device_t device;
    SCIBorrowDevice(sd, &device, device_id, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to borrow device: %x\n", err);
        return 1;
    }

    sci_remote_segment_t segment;
    SCIConnectDeviceMemory(sd, &segment, device, 0, 0, 0, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to connect to device BAR segment: %x\n", err);
        return 1;
    }
    
    // Map NVMe controller's BAR0
    sci_map_t map;
    volatile void* reg_ptr = SCIMapRemoteSegment(segment, &map, 0, 0x2000, NULL, SCI_FLAG_IO_MAP_IOSPACE, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to map device BAR segment\n");
        return 1;
    }

    if (run_master)
    {
        // Run as master
        nvm_ctrl_t ctrl;

        int status = nvm_init(&ctrl, device, reg_ptr);
        if (status != 0)
        {
            fprintf(stderr, "Failed to reset and initialize device: %s\n", strerror(status));
            return 2;
        }
        fprintf(stderr, "Controller initialized.\n");

        if (identify)
        {
            print_controller_info(&ctrl);
        }

        node_id = 0;
        SCIGetLocalNodeId(0, &node_id, 0, &err);

        sci_local_data_interrupt_t intr;
        SCICreateDataInterrupt(sd, &intr, 0, &intno, (sci_cb_data_interrupt_t) handle_command, &ctrl, SCI_FLAG_USE_CALLBACK | SCI_FLAG_FIXED_INTNO , &err);
        if (err != SCI_ERR_OK)
        {
            fprintf(stderr, "Failed to create data interrupt bla: %x\n", err);
            return 2;
        }
        
        fprintf(stderr, "Running master on node %u (interrupt %u)...\n", node_id, intno);

        signal(SIGINT, (sig_t) shutdown_master);
        signal(SIGTERM, (sig_t) shutdown_master);

        while (run_master)
        {
            sleep(1);
            // TODO: use condition variable instead
        }

        do
        {
            SCIRemoveDataInterrupt(intr, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

        // Clean up resources
        nvm_free(&ctrl);
    }
    else
    {
        // Run as slave
        int status = workload(device, node_id, intno, 1, (void*) reg_ptr, 0x2000);
        if (status != 0)
        {
            fprintf(stderr, "Workload failed\n");
        }
    }

    SCIUnmapSegment(map, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to unmap device BAR segment\n");
    }

    do
    {
        SCIDisconnectSegment(segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to disconnect BAR segment");
    }

    SCIReturnDevice(device, 0, &err);

    SCIClose(sd, 0, &err);
    return 0;
}

