#include "pci.h"
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


extern int cuda_workload(int ioctl_fd, const nvm_ctrl_t* ctrl, int dev, uint32_t ns, void* reg_ptr, size_t reg_len);


static struct option opts[] = {
    { "help", no_argument, NULL, 'h' },
    { "controller", required_argument, NULL, 'c' },
    { "cuda-device", required_argument, NULL, 'g' },
    { "identify", no_argument, NULL, 'i' },
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
    fprintf(stderr, "Usage: %s --controller=<pci bdf> [options...]\n", program_name);
}

static void show_help(const char* program_name)
{
    give_usage(program_name);
    fprintf(stderr,
            "\n"
            "    This program is intended to demonstrate how NVM submission and completion\n"
            "    queues can be hosted in memory on an Nvidia GPU. Note that this program\n"
            "    will do read and writes to blocks on the specified SSD disk and may ruin\n"
            "    any data stored there.\n"
            "\n"
            "  --controller=<pci bdf>      Specify the PCI BDF of the SSD disk to use.\n"
            "                              A BDF is on the format [xxxx:]xx:xx.x\n"
            "                              Use lspci -tv to find a suitable NVM controller.\n\n"
            "  --cuda-device=<device>      Specify CUDA device to use.\n"
            "                              Use nvidia-smi to identify available devices.\n\n"
           );
}


static int parse_bdf(const char* str, size_t len, uint64_t* bdf)
{
    char buffer[len + 1];
    strcpy(buffer, str);

    char* endptr;
    const char* sep = ":.";

    *bdf = 0UL;
    for (char* ptr = strtok(buffer, sep); ptr != NULL; ptr = strtok(NULL, sep))
    {
        uint32_t i = strtoul(ptr, &endptr, 16);
        
        if (endptr == NULL || *endptr != '\0' || i > 0xff)
        {
            return 1;
        }

        *bdf <<= 8UL;
        *bdf |= i;
    }

    return !(0UL < *bdf && *bdf <= 0xffffff1f07UL); // 0000:00:00.0 < bdf <= ffff:ff:1f.7
}


int main(int argc, char** argv)
{
    int opt;
    int idx;
    int err;

    int identify = 0;
    uint64_t bdf = 0;
    int cuda_device = 0;

    while ((opt = getopt_long(argc, argv, ":hc:g:i", opts, &idx)) != -1)
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

            case 'c': // set controller
                err = parse_bdf(optarg, strlen(optarg), &bdf);
                if (err != 0)
                {
                    fprintf(stderr, "Invalid PCI BDF string: %s\n", optarg);
                    return 'c';
                }
                break;

            case 'i': // identify controller
                identify = 1;
                break;

            case 'g': // set CUDA device
                // use cudaDeviceGetByPCIBusId(int* device, const char* pciBusId)
                break;

        }
    }

    int domain = (int) (bdf >> 24) & 0xffff;
    int bus = (int) (bdf >> 16) & 0xff;
    int slot = (int) (bdf >> 8) & 0x1f;
    int fun = (int) (bdf & 0x7);

    if (bus == 0 && slot == 0)
    {
        fprintf(stderr, "No NVM controller specified\n");
        give_usage(argv[0]);
        return 'c';
    }

    // Set up controller's config space as necessary
    err = pci_set_config(domain, bus, slot, fun);
    if (err != 0)
    {
        fprintf(stderr, "Failed to access device config space %04x:%02x:%02x.%x\n",
                domain, bus, slot, fun);
        return 2;
    }

    // Open communication channel with kernel module
    int ioctl_fd = open(CUNVME_PATH, O_SYNC | O_RDONLY);
    if (ioctl_fd < 0)
    {
        fprintf(stderr, "Failed to open ioctl file: %s\n", strerror(errno));
        return 1;
    }

    // Get descriptor to device's BAR0
    int reg_fd = pci_open_bar(domain, bus, slot, fun, 0);
    if (reg_fd < 0)
    {
        fprintf(stderr, "Failed to access device BAR resource file\n");
        close(ioctl_fd);
        return 2;
    }

    // Memory map device's BAR0
    volatile void* reg_ptr = mmap(NULL, 0x2000, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FILE, reg_fd, 0);
    if (reg_ptr == NULL)
    {
        fprintf(stderr, "Failed to mmap BAR resource file: %s\n", strerror(errno));
        close(reg_fd);
        close(ioctl_fd);
        return 2;
    }

    nvm_ctrl_t ctrl;

    // Reset and initialize controller
    fprintf(stderr, "Resetting controller %04x:%02x:%02x.%x...\n",
            domain, bus, slot, fun);

    err = nvm_init(&ctrl, ioctl_fd, reg_ptr);
    if (err != 0)
    {
        fprintf(stderr, "Failed to reset and initialize device: %s\n", strerror(err));
        munmap((void*) reg_ptr, 0x2000);
        close(reg_fd);
        close(ioctl_fd);
        return 2;
    }
    fprintf(stderr, "Controller initialized.\n");

    if (identify)
    {
        print_controller_info(&ctrl);
    }

    // Do CUDA workload to demonstrate queues hosted on GPU memory
    cuda_workload(ioctl_fd, &ctrl, cuda_device, 1, (void*) reg_ptr, 0x2000);

    // Clean up resources
    nvm_free(&ctrl, ioctl_fd);
    munmap((void*) reg_ptr, 0x2000);
    close(reg_fd);
    close(ioctl_fd);

    return 0;
}

