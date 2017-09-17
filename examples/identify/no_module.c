#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <getopt.h>
#include <stdbool.h>
#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_manager.h>
#include <nvm_rpc.h>
#include <nvm_dma.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "util.h"


/* 
 * Bus-device-function descriptor.
 * Used to identify a device/function in the PCI tree.
 */
struct bdf
{
    int     domain;
    int     bus;
    int     device;
    int     function;
};


static int lookup_ioaddrs(void* ptr, size_t page_size, size_t n_pages, uint64_t* ioaddrs)
{
    uint64_t vaddr = (uint64_t) ptr;
    size_t offset = (vaddr / page_size) * sizeof(void*);

    FILE* fp = fopen("/proc/self/pagemap", "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Failed to open page map: %s\n", strerror(errno));
        return errno;
    }

    if (fseek(fp, offset, SEEK_SET) != 0)
    {
        fclose(fp);
        fprintf(stderr, "Failed to seek: %s\n", strerror(errno));
        return errno;
    }

    if (fread(ioaddrs, sizeof(uint64_t), n_pages, fp) != n_pages)
    {
        fclose(fp);
        fprintf(stderr, "Failed to read: %s\n", strerror(errno));
        return errno;
    }

    fclose(fp);

    for (size_t i_page = 0; i_page < n_pages; ++i_page)
    {
        if (!(ioaddrs[i_page] & (1ULL << 63)))
        {
            fprintf(stderr, "Page not present in memory!\n");
            return EINVAL;
        }

        ioaddrs[i_page] = (ioaddrs[i_page] & ((1ULL << 54) - 1)) * page_size;
    }

    return 0;
}


static int execute_identify(nvm_ctrl_t ctrl, nvm_dma_t queues, void* ptr, uint64_t ioaddr)
{
    int status;
    nvm_manager_t mngr;
    nvm_rpc_t rpc;
    nvm_ctrl_info_t info;

    fprintf(stderr, "Resetting controller and setting up admin queues...\n");
    status = nvm_manager_register(&mngr, ctrl, queues);
    if (status != 0)
    {
        fprintf(stderr, "Failed to reset controller: %s\n", strerror(errno));
        return 1;
    }

    status = nvm_rpc_bind_local(&rpc, mngr);
    if (status != 0)
    {
        nvm_manager_unregister(mngr);
        fprintf(stderr, "Failed to create RPC handle: %s\n", strerror(errno));
        return 1;
    }

    status = nvm_rpc_ctrl_info(&info, rpc, ctrl, ptr, ioaddr);
    if (status != 0)
    {
        fprintf(stderr, "Failed to identify controller: %s\n", strerror(errno));
        status = 1;
        goto out;
    }

    print_ctrl_info(stdout, &info);

out:
    nvm_rpc_unbind(rpc);
    nvm_manager_unregister(mngr);
    return status;
}


static int identify_ctrl(nvm_ctrl_t ctrl)
{
    int status;
    void* queue_memory;
    nvm_dma_t window;
    uint64_t ioaddrs[3];

    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size == -1)
    {
        fprintf(stderr, "Failed to look up page size: %s\n", strerror(errno));
        return 1;
    }

    status = posix_memalign(&queue_memory, ctrl->page_size, 3 * page_size);
    if (status != 0)
    {
        fprintf(stderr, "Failed to allocate page-aligned memory: %s\n", strerror(status));
        return 1;
    }

    status = mlock(queue_memory, 3 * page_size);
    if (status != 0)
    {
        free(queue_memory);
        fprintf(stderr, "Failed to page-lock memory: %s\n", strerror(status));
        return 1;
    }

    status = lookup_ioaddrs(queue_memory, page_size, 3, ioaddrs);
    if (status != 0)
    {
        munlock(queue_memory, 3 * page_size);
        free(queue_memory);
        goto out;
    }

    status = nvm_dma_window_init(&window, ctrl, queue_memory, page_size, 3, ioaddrs);
    if (status != 0)
    {
        fprintf(stderr, "Failed to create DMA window: %s\n", strerror(status));
        status = 2;
        goto out;
    }

    status = execute_identify(ctrl, window, ((unsigned char*) queue_memory) + 2*page_size, ioaddrs[2]);

    nvm_dma_window_free(window);

out:
    munlock(queue_memory, 3 * page_size);
    free(queue_memory);
    return status;
}


static int pci_enable_device(const struct bdf* dev)
{
    char path[64];
    sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.%x/enable",
            dev->domain, dev->bus, dev->device, dev->function);

    FILE *fp = fopen(path, "w");
    if (fp == NULL)
    {
        fprintf(stderr, "Failed to open file descriptor: %s\n", strerror(errno));
        return errno;
    }

    fputc('1', fp);
    fclose(fp);
    return 0;
}


/*
 * Allow device to do DMA.
 */
static int pci_set_bus_master(const struct bdf* dev)
{
    char path[64];
    sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.%x/config", 
            dev->domain, dev->bus, dev->device, dev->function);

    FILE* fp = fopen(path, "r+");
    if (fp == NULL)
    {
        fprintf(stderr, "Failed to open config space file: %s\n", strerror(errno));
        return errno;
    }

    uint16_t command;
    fseek(fp, 0x04, SEEK_SET);
    fread(&command, sizeof(command), 1, fp);

    command |= (1 << 0x02);

    fseek(fp, 0x04, SEEK_SET);
    fwrite(&command, sizeof(command), 1, fp);

    fclose(fp);
    return 0;
}


/*
 * Open a file descriptor to device memory.
 */
static int pci_open_bar(const struct bdf* dev, int bar)
{
    char path[64];
    sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.%x/resource%d", 
            dev->domain, dev->bus, dev->device, dev->function, bar);

    int fd = open(path, O_RDWR);
    if (fd < 0)
    {
        fprintf(stderr, "Failed to open resource file: %s\n", strerror(errno));
    }

    return fd;
}


static void parse_args(int argc, char** argv, struct bdf* device);


int main(int argc, char** argv)
{
    int status;
    nvm_ctrl_t ctrl;

    struct bdf device;
    parse_args(argc, argv, &device);

    // Enable device
    status = pci_enable_device(&device);
    if (status != 0)
    {
        fprintf(stderr, "Failed to enable device %04x:%02x:%02x.%x\n",
                device.domain, device.bus, device.device, device.function);
        exit(1);
    }

    // Enable device DMA
    status = pci_set_bus_master(&device);
    if (status != 0)
    {
        fprintf(stderr, "Failed to access device config space %04x:%02x:%02x.%x\n",
                device.domain, device.bus, device.device, device.function);
        exit(2);
    }

    // Memory-map device memory
    int fd = pci_open_bar(&device, 0);
    if (fd < 0)
    {
        fprintf(stderr, "Failed to access device BAR memory\n");
        exit(3);
    }

    volatile void* ctrl_registers = mmap(NULL, NVM_CTRL_MEM_MINSIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FILE, fd, 0);
    if (ctrl_registers == NULL || ctrl_registers == MAP_FAILED)
    {
        fprintf(stderr, "Failed to memory map BAR reasource file: %s\n", strerror(errno));
        close(fd);
        exit(3);
    }

    // Get controller reference
    status = nvm_ctrl_init(&ctrl, ctrl_registers, NVM_CTRL_MEM_MINSIZE);
    if (status != 0)
    {
        munmap((void*) ctrl_registers, NVM_CTRL_MEM_MINSIZE);
        close(fd);
        fprintf(stderr, "Failed to get controller reference: %s\n", strerror(status));
    }

    status = identify_ctrl(ctrl);

    nvm_ctrl_free(ctrl);
    munmap((void*) ctrl_registers, NVM_CTRL_MEM_MINSIZE);
    close(fd);
    exit(status);
}


static void give_usage(const char* name)
{
    fprintf(stderr, "Usage: %s --ctrl=<pci bdf>\n", name);
}


static void show_help(const char* name)
{
    give_usage(name);
    fprintf(stderr, "    Create a manager and run an IDENTIFY CONTROLLER NVM admin command.\n\n"
            "    --ctrl     <pci bdf>       PCI bus-device-function to controller.\n"
            "    --help                     Show this information.\n");
}


static int parse_bdf(char* str, struct bdf* dev)
{
    char* colon = strrchr(str, ':');
    char* dot = strchr(colon != NULL ? colon : str, '.');
    char* endptr;

    char* function = "0";
    char* slot = str;
    char* bus = "0";
    char* domain = "0";

    if (colon != NULL)
    {
        bus = str;
        *colon++ = '\0';
        slot = colon;

        colon = strchr(str, ':');
        if (colon != NULL)
        {
            domain = str;
            *colon++ = '\0';
            bus = colon;
        }
    }

    if (dot != NULL)
    {
        *dot++ = '\0';
        function = dot;
    }

    dev->domain = strtoul(domain, &endptr, 16);
    if (endptr == NULL || *endptr != '\0' || dev->domain > 0xffff)
    {
        fprintf(stderr, "Invalid PCI domain number: '%s'\n", domain);
        return 1;
    }

    dev->bus = strtoul(bus, &endptr, 16);
    if (endptr == NULL || *endptr != '\0' || dev->bus > 0xff)
    {
        fprintf(stderr, "Invalid PCI bus number: '%s'\n", bus);
        return 1;
    }

    dev->device = strtoul(slot, &endptr, 16);
    if (endptr == NULL || *endptr != '\0' || dev->device > 0xff)
    {
        fprintf(stderr, "Invalid PCI device number: '%s'\n", slot);
        return 1;
    }

    dev->function = strtoul(function, &endptr, 0);
    if (endptr == NULL || *endptr != '\0')
    {
        fprintf(stderr, "Invalid PCI device function: '%s'\n", function);
        return 1;
    }

    return 0;
}



static void parse_args(int argc, char** argv, struct bdf* dev)
{
    // Command line options
    static struct option opts[] = {
        { "help", no_argument, NULL, 'h' },
        { "ctrl", required_argument, NULL, 'c' },
        { NULL, 0, NULL, 0 }
    };

    int opt;
    int idx;

    memset(dev, 0, sizeof(struct bdf));

    while ((opt = getopt_long(argc, argv, ":hc:", opts, &idx)) != -1)
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
                if (parse_bdf(optarg, dev) != 0)
                {
                    give_usage(argv[0]);
                    exit('c');
                }
                break;

            case 'h':
                show_help(argv[0]);
                exit(0);
        }
    }

    // Check if device actually exists
    char path[64];
    struct stat s;
    sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.%x",
            dev->domain, dev->bus, dev->device, dev->function);

    if (stat(path, &s) != 0 || ! S_ISDIR(s.st_mode))
    {
        fprintf(stderr, "%04x:%02x:%02x.%x is not a valid PCI device\n",
                dev->domain, dev->bus, dev->device, dev->function);
        exit('c');
    }
}

