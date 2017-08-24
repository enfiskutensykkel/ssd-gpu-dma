#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_manager.h>
#include <nvm_command.h>
#include <nvm_admin.h>
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


static int identify_controller(nvm_mngr_t manager, nvm_ctrl_info_t* features)
{
    memset(features, 0, sizeof(nvm_ctrl_info_t));

    int err = nvm_ctrl_get_info(manager, features);
    if (err != 0)
    {
        return err;
    }

    return 0;
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
    fprintf(stdout, "Serial Number           : %s\n", serial);
    fprintf(stdout, "Model Number            : %s\n", model);
    fprintf(stdout, "Maximum number of CQs   : %zu\n", info->max_cqs);
    fprintf(stdout, "Maximum number of SQs   : %zu\n", info->max_sqs);
    fprintf(stdout, "Number of namespaces    : %zu\n", info->n_ns);
}



static void give_usage(const char* program_name)
{
    fprintf(stderr, "Usage: %s --ctrl=<dev id> [--intno=<intr no>] [--adapt=<adapter no>] [--identify]\n", program_name);
}


static void show_help(const char* program_name)
{
    give_usage(program_name);
    fprintf(stderr, "\nRun NVME admin queue manager service.\n\n"
            "    --ctrl=<dev id>        Specify SmartIO device ID to NVME controller\n"
            "    --intno=<intr no>      Specify interrupt number to register\n"
            "    --adapt=<adapter no>   DIS adapter number\n"
            "    --identify             Print controller information\n"
            "\n");
}


int main(int argc, char** argv)
{
    int opt;
    int idx;
    char* endptr;

    static struct option opts[] = {
        { "help", no_argument, NULL, 'h' },
        { "identify", no_argument, NULL, 's' },
        { "show", no_argument, NULL, 's' },
        { "controller", required_argument, NULL, 'c' },
        { "ctrl", required_argument, NULL, 'c' },
        { "device", required_argument, NULL, 'c' },
        { "dev", required_argument, NULL, 'c' },
        { "interrupt", required_argument, NULL, 'i' },
        { "intr", required_argument, NULL, 'i' },
        { "intno", required_argument, NULL, 'i' },
        { "adapter", required_argument, NULL, 'a' },
        { "adapt", required_argument, NULL, 'a' },
        { NULL, 0, NULL, 0}
    };

    bool identify = false;
    uint64_t device_id = 0;
    uint32_t interrupt_no = 0;
    uint32_t adapter = 0;
    uint32_t unique_id = 0;

    // Parse command line arguments
    while ((opt = getopt_long(argc, argv, ":hsc:i:a:", opts, &idx)) != -1)
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

            case 's': // identify controller
                identify = true;
                break;

            case 'c': // device identifier
                endptr = NULL;
                device_id = strtoul(optarg, &endptr, 0);

                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid device id: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('c');
                }
                break;

            case 'i': // interrupt number
                endptr = NULL;
                interrupt_no = strtoul(optarg, &endptr, 0);

                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid interrupt number: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('i');
                }
                break;

            case 'a': // adapter number
                endptr = NULL;
                adapter = strtoul(optarg, &endptr, 0);

                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid adapter number: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('a');
                }
                break;
        }
    }

    if (device_id == 0)
    {
        fprintf(stderr, "No controller specified!\n");
        give_usage(argv[0]);
        exit('c');
    }

    sci_error_t status;
    SCIInitialize(0, &status);
    if (status != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to initialize SISCI: %s\n", SCIGetErrorString(status));
        exit(2);
    }

    uint32_t node_id = 0;
    SCIGetLocalNodeId(adapter, &node_id, 0, &status);
    if (status != SCI_ERR_OK)
    {
        fprintf(stderr, "Unexpected error while getting local node id: %s\n", SCIGetErrorString(status));
    }

    fprintf(stdout, "Reading controller registers...");
    fflush(stdout);

    nvm_ctrl_t controller;
    int err = nvm_ctrl_init(&controller, device_id, adapter);
    if (err != 0)
    {
        fprintf(stdout, "FAIL\n");

        fprintf(stderr, "Failed to initialize NVM controller: %s\n", strerror(err));
        exit(1);
    }

    fprintf(stdout, "DONE\n");

    fprintf(stdout, "Resetting controller...........");
    fflush(stdout);

    nvm_mngr_t manager;
    err = nvm_mngr_init(&manager, &controller, unique_id);
    if (err != 0)
    {
        fprintf(stdout, "FAIL\n");

        nvm_ctrl_free(&controller);
        fprintf(stderr, "Failed to initialize NVM manager: %s\n", strerror(err));
        exit(1);
    }

    fprintf(stdout, "DONE\n");

    fprintf(stdout, "Identifying controller.........");
    fflush(stdout);

    nvm_ctrl_info_t features;
    if (identify)
    {
        err = identify_controller(manager, &features);
        fprintf(stdout, err == 0 ? "DONE\n" : "FAIL\n");
    }
    else
    {
        fprintf(stdout, "SKIP\n");
    }

    if (identify)
    {
        print_controller_info(&features);
    }

    fprintf(stdout, "Running NVME admin manager on node %u (interrupt %u)...\n", node_id, interrupt_no);


    nvm_mngr_free(manager);
    nvm_ctrl_free(&controller);

    fprintf(stdout, "kthxbye\n");

    SCITerminate();
    exit(0);
}

