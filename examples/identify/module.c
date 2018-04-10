#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_dma.h>
#include <nvm_aq.h>
#include <nvm_admin.h>
#include <nvm_util.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <getopt.h>
#include <string.h>
#include <errno.h>
#include "common.h"



static void parse_args(int argc, char** argv, char** device, uint32_t* ns_id);



static int open_fd(const char* path)
{
    int fd;

    fd = open(path, O_RDWR|O_NONBLOCK);
    if (fd < 0)
    {
        fprintf(stderr, "Failed to open descriptor: %s\n", strerror(errno));
        return -1;
    }

    return fd;
}


int main(int argc, char** argv)
{
    int status;
    nvm_ctrl_t* ctrl;
    nvm_dma_t* window = NULL;
    nvm_aq_ref admin = NULL;
    uint32_t nvm_namespace;
    void* memory;

    long page_size = sysconf(_SC_PAGESIZE);

    char* path = NULL;
    parse_args(argc, argv, &path, &nvm_namespace);

    int fd = open_fd(path);
    if (fd < 0)
    {
        exit(1);
    }

    status = nvm_ctrl_init(&ctrl, fd);
    if (status != 0)
    {
        close(fd);
        fprintf(stderr, "Failed to get controller reference: %s\n", strerror(status));
        exit(1);
    }

    close(fd);

    status = posix_memalign(&memory, ctrl->page_size, 3 * page_size);
    if (status != 0)
    {
        fprintf(stderr, "Failed to allocate page-aligned memory: %s\n", strerror(status));
        nvm_ctrl_free(ctrl);
        exit(2);
    }

    status = nvm_dma_map_host(&window, ctrl, memory, 3 * page_size);
    if (status != 0)
    {
        free(memory);
        nvm_ctrl_free(ctrl);
        exit(1);
    }

    admin = reset_ctrl(ctrl, window);
    if (admin == NULL)
    {
        status = 1;
        goto leave;
    }

    status = identify_ctrl(admin, NVM_DMA_OFFSET(window, 2), window->ioaddrs[2]);
    if (status != 0)
    {
        goto leave;
    }

    if (nvm_namespace != 0)
    {
        status = identify_ns(admin, nvm_namespace, NVM_DMA_OFFSET(window, 2), window->ioaddrs[2]);
    }

leave:
    nvm_aq_destroy(admin);
    nvm_dma_unmap(window);
    free(memory);
    nvm_ctrl_free(ctrl);    

    fprintf(stderr, "Goodbye!\n");
    exit(status);
}


static void give_usage(const char* name)
{
    fprintf(stderr, "Usage: %s --ctrl=<dev id>\n", name);
}


static void show_help(const char* name)
{
    give_usage(name);
    fprintf(stderr, "\nCreate a manager and run an IDENTIFY CONTROLLER NVM admin command.\n\n"
            "    --ctrl     <path>          Path to controller device (/dev/libnvmXXX).\n"
            "    --ns       <namespace>     Show information about NVM namespace.\n"
            "    --help                     Show this information.\n\n");
}


static void parse_args(int argc, char** argv, char** dev, uint32_t* ns_id)
{
    static struct option opts[] = {
        { "help", no_argument, NULL, 'h' },
        { "ctrl", required_argument, NULL, 'c' },
        { "ns", required_argument, NULL, 'n' },
        { NULL, 0, NULL, 0 }
    };

    int opt;
    int idx;
    char* endptr = NULL;

    *dev = NULL;
    *ns_id = 0;

    while ((opt = getopt_long(argc, argv, ":hc:n:", opts, &idx)) != -1)
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
                *dev = optarg;
                break;

            case 'n': // namespace identifier
                *ns_id = strtoul(optarg, &endptr, 0);
                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid NVM namespace");
                    give_usage(argv[0]);
                    exit('n');
                }
                break;

            case 'h':
                show_help(argv[0]);
                exit(0);
        }
    }

    if (*dev == NULL)
    {
        fprintf(stderr, "Controller is not set!\n");
        give_usage(argv[0]);
        exit('c');
    }
}
