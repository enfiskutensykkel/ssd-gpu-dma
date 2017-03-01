#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

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

    // Open file descriptor to device's BAR0 resource file
    strncat(path, "/resource0", sizeof(path) - strlen(path) - 1);
    int bar0_fd = open(path, O_RDWR);
    if (bar0_fd < 0)
    {
        fprintf(stderr, "Couldn't open resource file: %s\n", strerror(errno));
        return 3;
    }

    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size != PAGE_SIZE)
    {
        fprintf(stderr, "WARNING: System page size is not 4096 bytes\n");
    }

    // Memory map resource file
    volatile void* bar0_ptr = mmap(NULL, 2 * PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, bar0_fd, 0);
    if (bar0_ptr == NULL)
    {
        fprintf(stderr, "Failed to mmap: %s\n", strerror(errno));
        close(bar0_fd);
        return 3;
    }

    close(bar0_fd);
    return 0;
}
