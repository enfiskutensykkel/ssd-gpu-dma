#include "nvm/types.h"
#include "nvm/ctrl.h"
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


int workload(nvm_ctrl_t* ctrl, uint32_t ns, void* reg_ptr, size_t reg_len);


static struct option opts[] = {
    { "help", no_argument, NULL, 'h' },
    { "identify", no_argument, NULL, 'i' },
    { "device", 1, NULL, 'd' },
    { NULL, 0, NULL, 0 }
};


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
}


static void give_usage(const char* program_name)
{
    fprintf(stderr, "Usage: %s --device=<dev id> [options...]\n"
                    "   or: %s --remote=<node id> [options...\n", 
                    program_name,
                    program_name);
}

static void show_help(const char* program_name)
{
    give_usage(program_name);
}

//static int parse_bdf(const char* str, size_t len, uint64_t* bdf)
//{
//    char buffer[len + 1];
//    strcpy(buffer, str);
//
//    char* endptr;
//    const char* sep = ":.";
//
//    *bdf = 0UL;
//    for (char* ptr = strtok(buffer, sep); ptr != NULL; ptr = strtok(NULL, sep))
//    {
//        uint32_t i = strtoul(ptr, &endptr, 16);
//        
//        if (endptr == NULL || *endptr != '\0' || i > 0xff)
//        {
//            return 1;
//        }
//
//        *bdf <<= 8UL;
//        *bdf |= i;
//    }
//
//    return !(0UL < *bdf && *bdf <= 0xffffff1f07UL); // 0000:00:00.0 < bdf <= ffff:ff:1f.7
//}


int main(int argc, char** argv)
{
    int opt;
    int idx;
    int err;
    sci_error_t scierr;
    char* endptr;

    int identify = 0;
    uint32_t remote_node = 0;
    uint64_t device_id = 0;

    while ((opt = getopt_long(argc, argv, ":hid:r:", opts, &idx)) != -1)
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

            case 'i': // identify controller
                identify = 1;
                break;

            case 'r': // connect to remote host
                endptr = NULL;
                remote_node = strtoul(optarg, &endptr, 0);

                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Not a valid node id: %s\n", optarg);
                    give_usage(argv[0]);
                    return 'r';
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

    SCIInitialize(0, &scierr);

    sci_desc_t sd;
    SCIOpen(&sd, 0, &scierr);

    sci_device_t device;
    SCIBorrowDevice(sd, &device, device_id, 0, &scierr);
    if (scierr != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to borrow device: %x\n", scierr);
        return 1;
    }

    sci_remote_segment_t segment;
    SCIConnectDeviceMemory(sd, &segment, device, 0, 0, 0, 0, &scierr);
    if (scierr != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to connect to device BAR: %x\n", scierr);
        return 1;
    }

    sci_map_t map;
    volatile void* reg_ptr = SCIMapRemoteSegment(segment, &map, 0, 0x2000, NULL, SCI_FLAG_IO_MAP_IOSPACE, &scierr);
    if (scierr != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to map BAR segment\n");
        return 1;
    }

    // Reset and initialize controller
    nvm_ctrl_t ctrl;

    fprintf(stderr, "Resetting controller...\n");

    err = nvm_init(&ctrl, device, reg_ptr);
    if (err != 0)
    {
        fprintf(stderr, "Failed to reset and initialize device: %s\n", strerror(err));
        return 2;
    }
    fprintf(stderr, "Controller initialized.\n");

    if (identify)
    {
        print_controller_info(&ctrl);
    }

    err = workload(&ctrl, 1, (void*) reg_ptr, 0x2000);
    if (err != 0)
    {
        fprintf(stderr, "Workload failed: %s\n", strerror(err));
    }

    // Clean up resources
    nvm_free(&ctrl);

    SCIUnmapSegment(map, 0, &scierr);
    if (scierr != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to unmap BAR segment\n");
        return 1;
    }

    do
    {
        SCIDisconnectSegment(segment, 0, &scierr);
    }
    while (scierr == SCI_ERR_BUSY);

    SCIReturnDevice(device, 0, &scierr);

    SCIClose(sd, 0, &scierr);
    return 0;
}

