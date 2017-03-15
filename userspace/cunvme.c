#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "nvme.h"
#include "nvme_init.h"
#include "cuda.h"


/* Offset to the COMMAND register in config space */
#define CONFIG_COMMAND  0x04



static FILE* open_config_space(char* path)
{
    char filename[256];
    filename[0] = '\0';
    strncpy(filename, path, sizeof(filename));
    strncat(filename, "/config", sizeof(filename) - strlen(filename) - 1);
    return fopen(filename, "r+");
}


static void read_register(FILE* fptr, size_t offset, size_t size, void* ptr)
{
    fseek(fptr, offset, SEEK_SET);
    fread(ptr, 1, size, fptr);
}


static void write_register(FILE* fptr, size_t offset, size_t size, void* ptr)
{
    fseek(fptr, offset, SEEK_SET);
    fwrite(ptr, 1, size, fptr);
}


static void print_controller_info(nvm_controller_t controller)
{
    unsigned char* data = controller->data.virt_addr;

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


int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <domain>:<bus>:<device>.<fn>\n", argv[0]);
        return 1;
    }

    char path[256];
    path[0] = '\0';
    strcpy(path, "/sys/bus/pci/devices/");
    strncat(path, argv[1], sizeof(path) - strlen(path) - 1);
    
    // Set Bus Master Enable to enable DMA for controller
    FILE* config_fd = open_config_space(path);
    if (config_fd == NULL)
    {
        fprintf(stderr, "Couldn't open config space file: %s\n", strerror(errno));
        return 2;
    }

    uint16_t command;
    read_register(config_fd, CONFIG_COMMAND, 2, &command);
    command |= 1 << 2;
    write_register(config_fd, CONFIG_COMMAND, 2, &command);

    fclose(config_fd);

    // Open communication channel with kernel module
    int ioctl_fd = open(CUNVME_PATH, O_SYNC | O_RDONLY);
    if (ioctl_fd < 0)
    {
        fprintf(stderr, "Couldn't open ioctl file: %s\n", strerror(errno));
        return 3;
    }

    // Open file descriptor to device's BAR0 resource file
    strncat(path, "/resource0", sizeof(path) - strlen(path) - 1);
    int bar0_fd = open(path, O_RDWR);
    if (bar0_fd < 0)
    {
        fprintf(stderr, "Couldn't open resource file: %s\n", strerror(errno));
        close(ioctl_fd);
        return 3;
    }

    // Memory map resource file
    volatile void* register_mem = mmap(NULL, 0x1000 + MAX_DBL_MEM, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FILE, bar0_fd, 0);
    if (register_mem == NULL)
    {
        fprintf(stderr, "Failed to mmap: %s\n", strerror(errno));
        close(ioctl_fd);
        close(bar0_fd);
        return 3;
    }

    nvm_controller_t handle;

    // Initialize controller
    fprintf(stderr, "Resetting device...\n");
    int status = nvm_init(&handle, ioctl_fd, register_mem, MAX_DBL_MEM);
    if (status != 0)
    {
        fprintf(stderr, "Failed to reset and initialize device: %s\n", strerror(status));
        munmap((void*) register_mem, 0x1000 + MAX_DBL_MEM);
        close(ioctl_fd);
        close(bar0_fd);
        return 4;
    }

    // Print some info about the controller
    print_controller_info(handle);

    // Do CUDA work load
    status = start_kernel(ioctl_fd, -1, handle);
    if (status != 0)
    {
        fprintf(stderr, "Failed\n");
    }

    // Clean up resources
    nvm_free(handle, ioctl_fd);
    munmap((void*) register_mem, 0x1000 + MAX_DBL_MEM);
    close(ioctl_fd);
    close(bar0_fd);

    return status;
}
