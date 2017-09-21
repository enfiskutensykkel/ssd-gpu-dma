#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_dma.h>
#include <nvm_manager.h>
#include <nvm_rpc.h>
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
#include "util.h"


static void parse_args(int argc, char** argv, uint64_t* device_id);


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


int main(int argc, char** argv)
{
    int status;
    nvm_ctrl_t ctrl;
    nvm_dma_t window;
    void* memory;

    long page_size = sysconf(_SC_PAGESIZE);

    uint64_t dev_id;
    parse_args(argc, argv, &dev_id);

    status = nvm_ctrl_init(&ctrl, dev_id);
    if (status != 0)
    {
        fprintf(stderr, "Failed to get controller reference: %s\n", strerror(status));
        exit(1);
    }

    status = posix_memalign(&memory, ctrl->page_size, 3 * page_size);
    if (status != 0)
    {
        fprintf(stderr, "Failed to allocate page-aligned memory: %s\n", strerror(status));
        nvm_ctrl_free(ctrl);
        exit(2);
    }

    status = nvm_dma_window_host_mem(&window, ctrl, memory, 3 * page_size);
    if (status != 0)
    {
        free(memory);
        nvm_ctrl_free(ctrl);
        exit(1);
    }

    status = execute_identify(ctrl, window, ((unsigned char*) memory) + 2 * page_size, window->ioaddrs[2]);

    nvm_dma_window_free(window);
    free(memory);
    nvm_ctrl_free(ctrl);    

    fprintf(stderr, "Goodbye!\n");
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



static void parse_args(int argc, char** argv, uint64_t* dev)
{
    static struct option opts[] = {
        { "help", no_argument, NULL, 'h' },
        { "ctrl", required_argument, NULL, 'c' },
        { NULL, 0, NULL, 0 }
    };

    int opt;
    int idx;

    bool dev_set = false;
    *dev = 0;

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
                dev_set = true;
                if (parse_u64(optarg, dev, 10) != 0)
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

    if (!dev_set)
    {
        fprintf(stderr, "Device ID is not set!\n");
        give_usage(argv[0]);
        exit('c');
    }
}
