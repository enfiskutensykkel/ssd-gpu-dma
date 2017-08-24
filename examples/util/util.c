#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_manager.h>
#include <nvm_command.h>
#include <nvm_admin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <getopt.h>
#include <unistd.h>
#include <sisci_types.h>
#include <sisci_api.h>


struct queue_memory
{
    sci_desc_t              sd;
    sci_local_segment_t     segment;
    sci_map_t               map;
};


static void* create_segment(uint32_t id, uint32_t adapter, struct queue_memory* mem)
{
    long page_size = sysconf(_SC_PAGESIZE);

    sci_error_t err;

    SCIOpen(&mem->sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to open virtual device: %s\n", SCIGetErrorString(err));
        return NULL;
    }

    SCICreateSegment(mem->sd, &mem->segment, id, 2 * page_size, NULL, NULL, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to create segment: %s\n", SCIGetErrorString(err));
        goto close;
    }

    SCIPrepareSegment(mem->segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to prepare segment: %s\n", SCIGetErrorString(err));
        goto remove;
    }

    void* ptr = SCIMapLocalSegment(mem->segment, &mem->map, 0, 2 * page_size, NULL, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to map segment: %s\n", SCIGetErrorString(err));
        goto remove;
    }

    SCISetSegmentAvailable(mem->segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to set available: %s\n", SCIGetErrorString(err));
        goto unmap;
    }

    memset(ptr, 0, 2 * page_size);
    return ptr;

unmap:
    do
    {
        SCIUnmapSegment(mem->map, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

remove:
    do
    {
        SCIRemoveSegment(mem->segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

close:
    SCIClose(mem->sd, 0, &err);
    return NULL;
}


static void completion_queue_cmd(nvm_cmd_t* cmd, uint64_t ioaddr, uint16_t queue_id)
{
    long page_size = sysconf(_SC_PAGESIZE);

    nvm_cmd_header(cmd, NVM_ADMIN_CREATE_COMPLETION_QUEUE, 0);
    nvm_cmd_data_ptr(cmd, ioaddr, 0);

    cmd->dword[10] = ((page_size / sizeof(nvm_cmd_t)) << 16) | queue_id;
    cmd->dword[11] = (0x0000 << 16) | (0x00 << 1) | 0x01;
}



static void give_usage(const char* program_name)
{
    fprintf(stderr, "Usage: %s --ctrl=<dev id> --node=<node id> --intno=<intr no> [--adapt=<adapter no>] [--identify]\n", program_name);
}


static void show_help(const char* program_name)
{
    give_usage(program_name);
}


int main(int argc, char** argv)
{
    int opt;
    int idx;
    char* endptr;

    static struct option opts[] = {
        { "help", no_argument, NULL, 'h' },
        { "controller", required_argument, NULL, 'c' },
        { "ctrl", required_argument, NULL, 'c' },
        { "remote-interrupt", required_argument, NULL, 'i' },
        { "remote-intr", required_argument, NULL, 'i' },
        { "remote-intno", required_argument, NULL, 'i' },
        { "interrupt", required_argument, NULL, 'i' },
        { "intr", required_argument, NULL, 'i' },
        { "intno", required_argument, NULL, 'i' },
        { "adapter", required_argument, NULL, 'a' },
        { "adapt", required_argument, NULL, 'a' },
        { "remote-node", required_argument, NULL, 'r' },
        { "remote", required_argument, NULL, 'r' },
        { "node-id", required_argument, NULL, 'r' },
        { "node", required_argument, NULL, 'r' },
        { "rn", required_argument, NULL, 'r' },
        { NULL, 0, NULL, 0 }
    };

    uint64_t device_id = 0;
    uint32_t adapter = 0;
    uint32_t remote_intr = 0;
    uint32_t remote_node = 0;

    while ((opt = getopt_long(argc, argv, ":hc:i:a:r:", opts, &idx)) != -1)
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
                remote_intr = strtoul(optarg, &endptr, 0);

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

            case 'r': // adapter number
                endptr = NULL;
                remote_node = strtoul(optarg, &endptr, 0);

                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid node id: %s\n", optarg);
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
    else if (remote_node == 0)
    {
        fprintf(stderr, "No remote node id specified!\n");
        give_usage(argv[0]);
        exit('r');
    }

    sci_error_t status;
    SCIInitialize(0, &status);
    if (status != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to initialize SISCI: %s\n", SCIGetErrorString(status));
        exit(2);
    }

    uint32_t queue_id = 0;
    SCIGetLocalNodeId(adapter, &queue_id, 0, &status);
    if (status != SCI_ERR_OK)
    {
        exit(2);
    }

    nvm_ctrl_t ctrl;
    int err = nvm_ctrl_init(&ctrl, device_id, adapter);
    if (err != 0)
    {
        fprintf(stderr, "Failed to initialize controller: %s\n", strerror(err));
        exit(1);
    }

    struct queue_memory queue_mem;
    void* queue_ptr = create_segment(0x1000, adapter, &queue_mem);
    if (queue_ptr == NULL)
    {
        fprintf(stderr, "Failed to create queue memory\n");
        nvm_ctrl_free(&ctrl);
        exit(3);
    }

    uint64_t ioaddr;
    err = nvm_dma_window_create(queue_mem.segment, &ctrl, &ioaddr);
    if (err != 0)
    {
        fprintf(stderr, "Failed to map queue memory for device: %s\n", strerror(err));
        nvm_ctrl_free(&ctrl);
        exit(3);
    }

    nvm_cmd_t cmd;
    completion_queue_cmd(&cmd, ioaddr, queue_id);

    nvm_cpl_t cpl;
    err = nvm_remote_cmd(remote_node, adapter, remote_intr, &cmd, &cpl, 10000);
    if (err != 0)
    {
        fprintf(stderr, "buuuuuu\n");
        exit(3);
    }

    submission_queue_cmd(&cmd, ioaddr + sysconf(_SC_PAGESIZE), queue_id, queue_id + 1);

    nvm_ctrl_free(&ctrl);

    SCITerminate();
    exit(0);
}
