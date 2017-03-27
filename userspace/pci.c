#include "pci.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>


#define CONFIG_COMMAND      0x04    // Offset to the COMMAND register in config space
#define BUS_MASTER          0x02    // Offset to the BusMaster bit in COMMAND register


static ssize_t read_register(FILE* fptr, size_t offset, size_t size, void* ptr)
{
    fseek(fptr, offset, SEEK_SET);
    return fread(ptr, 1, size, fptr);
}


static ssize_t write_register(FILE* fptr, size_t offset, size_t size, void* ptr)
{
    fseek(fptr, offset, SEEK_SET);
    return fwrite(ptr, 1, size, fptr);
}


int pci_set_config(int domain, int bus, int dev, int fun)
{
    char path[64];
    sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.%x/config", domain, bus, dev, fun);

    FILE* config_fp = fopen(path, "r+");
    if (config_fp == NULL)
    {
        fprintf(stderr, "Failed to open config space file: %s\n", strerror(errno));
        return errno;
    }

    uint16_t command;
    read_register(config_fp, CONFIG_COMMAND, sizeof(uint16_t), &command);
    command |= (1 << BUS_MASTER);
    write_register(config_fp, CONFIG_COMMAND, sizeof(uint16_t), &command);

    fclose(config_fp);
    return 0;
}


int pci_open_bar(int domain, int bus, int dev, int fun, int bar)
{
    char path[64];
    sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.%x/resource%d", domain, bus, dev, fun, bar);

    int fd = open(path, O_RDWR);
    if (fd < 0)
    {
        fprintf(stderr, "Failed to open resource file: %s\n", strerror(errno));
    }

    return fd;
}
