#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_dma.h>
#include <nvm_manager.h>
#include <nvm_rpc.h>
#include <nvm_command.h>
#include <nvm_util.h>
#include <nvm_queue.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <sisci_api.h>
#include <time.h>
#include "dis/segment.h"
#include "dis/node.h"
#include "util.h"


struct cl_args
{
    bool        identify_ctrl;
    bool        use_ascii;
    bool        verbose;
    uint64_t    smartio_dev_id;
    uint32_t    ctrl_adapter;
    uint32_t    namespace;
    size_t      start_lba;
    size_t      offset;
    size_t      length;
    char*       data;
};


static void give_usage(const char* program_name);
static void show_help(const char* program_name);
static void parse_opts(int argc, char** argv, struct cl_args* args);


static int identify_controller(nvm_rpc_t rpc, nvm_ctrl_t ctrl, uint32_t adapter, nvm_ctrl_info_t* info)
{
    int status;
    struct segment segment;
    nvm_dma_t identify_wnd;

    status = segment_create(&segment, random_id(), 0x1000);
    if (status != 0)
    {
        fprintf(stderr, "Failed to create segment: %s\n", strerror(status));
        return status;
    }

    status = dma_create(&identify_wnd, ctrl, &segment, adapter);
    if (status != 0)
    {
        segment_remove(&segment);
        fprintf(stderr, "Failed to create DMA window: %s\n", strerror(status));
        return status;
    }

    status = nvm_rpc_ctrl_info(info, rpc, ctrl, identify_wnd->vaddr, identify_wnd->ioaddrs[0]);
    if (status != 0)
    {
        fprintf(stderr, "Failed to identify controller: %s\n", strerror(status));
    }

    dma_remove(&identify_wnd, &segment, adapter);
    segment_remove(&segment);

    return status;
}


static int identify_ns(nvm_rpc_t rpc, nvm_ctrl_t ctrl, uint32_t adapter, uint32_t ns_id, nvm_ns_info_t* info)
{
    int status;
    struct segment segment;
    nvm_dma_t identify_wnd;

    status = segment_create(&segment, random_id(), 0x1000);
    if (status != 0)
    {
        fprintf(stderr, "Failed to create segment: %s\n", strerror(status));
        return status;
    }

    status = dma_create(&identify_wnd, ctrl, &segment, adapter);
    if (status != 0)
    {
        segment_remove(&segment);
        fprintf(stderr, "Failed to create DMA window: %s\n", strerror(status));
        return status;
    }

    status = nvm_rpc_ns_info(info, rpc, ns_id, identify_wnd->vaddr, identify_wnd->ioaddrs[0]);
    if (status != 0)
    {
        fprintf(stderr, "Failed to identify namespace: %s\n", strerror(status));
    }

    dma_remove(&identify_wnd, &segment, adapter);
    segment_remove(&segment);

    return status;
}


static size_t dequeue_cpls(nvm_queue_t* cq, nvm_queue_t* sq, uint64_t timeout)
{
    nvm_cpl_t* cpl;
    size_t n_cpls = 0;
    
    while ((cpl = cq_dequeue_block(cq, timeout)) != NULL)
    {
        sq_update(sq, cpl);
        ++n_cpls;
    }

    return n_cpls;
}


static void dump_memory(FILE* fp, void* vaddr, size_t size, bool ascii)
{
    size_t i;
    size_t skip = 0;
    size_t j = 0;
    size_t lbreak = 64;

    for (i = 0; i < 64; ++i)
    {
        fputc('=', fp);
    }
    fprintf(fp, "\n");

    for (i = 0; i < size; ++i)
    {
        unsigned char byte = *(((uint8_t*) vaddr) + i);
        if (byte != 0)
        {
            if (skip > 0)
            {
                fprintf(fp, "\n...skipped %zu bytes\n", skip);
                skip = 0;
                j = 0;
            }

            fprintf(fp, ascii ? "%c" : "%02x ", byte);
            if (++j == lbreak)
            {
                fprintf(fp, "\n");
                j = 0;
            }
        }
        else
        {
            ++skip;
        }
    }

    if (skip > 0)
    {
        fprintf(fp, "\n...skipped %zu bytes", skip);
    }
    fprintf(fp, "\n");
    for (i = 0; i < 64; ++i)
    {
        fputc('=', fp);
    }
    fprintf(fp, "\n");
}


static size_t set_data_pointer(nvm_cmd_t* cmd, size_t page_size, size_t transfer_size, nvm_dma_t prp, size_t buffer_offset, nvm_dma_t buffer)
{
    // Set data pointer directly if possible
    if (transfer_size <= page_size)
    {
        nvm_cmd_data_ptr(cmd, buffer->ioaddrs[buffer_offset], 0);
        return buffer_offset + 1;
    }
    else if (transfer_size <= 2 * page_size)
    {
        nvm_cmd_data_ptr(cmd, buffer->ioaddrs[buffer_offset], buffer->ioaddrs[buffer_offset + 1]);
        return buffer_offset + 2;
    }

    // Create PRP list in memory
    size_t pages_used = nvm_prp_list(prp->vaddr, page_size, transfer_size, prp->ioaddrs, &buffer->ioaddrs[buffer_offset + 1]);
    nvm_cmd_data_ptr(cmd, buffer->ioaddrs[buffer_offset], prp->ioaddrs[0]);
    return buffer_offset + pages_used + 1;
}



static int transfer(nvm_ctrl_info_t* ctrl, nvm_ns_info_t* ns, nvm_queue_t* cq, nvm_queue_t* sq, nvm_dma_t prp_list, nvm_dma_t buffer, size_t transfer_size, struct cl_args* args)
{
    nvm_cmd_t* cmd;
    uint64_t start_lba = args->start_lba;
    size_t i_buffer_page = 0;
    size_t page_size = ctrl->page_size;
    size_t blk_size = ns->lba_data_size;
    uint32_t ns_id = ns->ns_id;
    size_t n_cmds = 0;

    uint8_t opcode = args->data == NULL ? NVM_IO_READ : NVM_IO_WRITE;

    // Divide transfer into transfer_size sized chunks
    while (i_buffer_page < buffer->n_ioaddrs)
    {
        size_t curr_transfer_size = _MIN(transfer_size, blk_size * 0x10000);
        uint64_t n_blks = DMA_SIZE(curr_transfer_size, blk_size) / blk_size;

        cmd = sq_enqueue(sq);
        if (cmd == NULL)
        {
            // Queue is full, wait for some time
            size_t n_cpls = dequeue_cpls(cq, sq, ctrl->timeout);
            if (n_cpls > 0)
            {
                cq_update(cq);
                n_cmds -= n_cpls;
            }
            continue;
        }

        i_buffer_page = set_data_pointer(cmd, page_size, curr_transfer_size, prp_list, i_buffer_page, buffer);

        // Create rest of command
        nvm_cmd_header(cmd, opcode, ns_id);
        cmd->dword[10] = start_lba;
        cmd->dword[12] = (n_blks - 1) & 0xffff;
        start_lba += n_blks;

        // Submit command
        sq_submit(sq);
        ++n_cmds;
    }

    if (n_cmds > 0)
    {
        cq_dequeue_block(cq, ctrl->timeout);
    }

    if (args->data == NULL)
    {
        dump_memory(stdout, buffer->vaddr, args->length, args->use_ascii);
    }

    return 0;
}


static int start_transfer(nvm_rpc_t rpc, nvm_ctrl_t ctrl, struct cl_args* args, nvm_queue_t* cq, nvm_queue_t* sq, struct segment* buffer)
{
    int status;
    struct segment prp_list;
    nvm_dma_t prp_wnd;
    nvm_dma_t rw_wnd;
    nvm_ctrl_info_t ctrl_info;
    nvm_ns_info_t ns_info;

    // Get controller information
    status = identify_controller(rpc, ctrl, args->ctrl_adapter, &ctrl_info);
    if (status != 0)
    {
        return 2;
    }

    // Get namespace information
    status = identify_ns(rpc, ctrl, args->ctrl_adapter, args->namespace, &ns_info);
    if (status != 0)
    {
        return 2;
    }

    status = dma_create(&rw_wnd, ctrl, buffer, args->ctrl_adapter);
    if (status != 0)
    {
        fprintf(stderr, "Failed to create read/write window: %s\n", strerror(status));
        return 2;
    }

    size_t transfer_size = _MIN(ctrl_info.max_transfer_size, args->length);
    size_t n_prp_pages = nvm_num_prp_pages(ctrl->page_size, transfer_size);

    status = segment_create(&prp_list, random_id(), n_prp_pages * ctrl->page_size);
    if (status != 0)
    {
        dma_remove(&rw_wnd, buffer, args->ctrl_adapter);
        fprintf(stderr, "Failed to create PRP list: %s\n", strerror(status));
        return 1;
    }

    status = dma_create(&prp_wnd, ctrl, &prp_list, args->ctrl_adapter);
    if (status != 0)
    {
        dma_remove(&rw_wnd, buffer, args->ctrl_adapter);
        segment_remove(&prp_list);
        fprintf(stderr, "Failed to create PRP window: %s\n", strerror(status));
        return 2;
    }

    if (args->verbose)
    {
        fprintf(stderr, "Namespace block size  : %zu\n", ns_info.lba_data_size);
        fprintf(stderr, "Total transfer size   : %zu\n", args->length);
        fprintf(stderr, "Max transfer data size: %zu\n", ctrl_info.max_transfer_size);
        fprintf(stderr, "Start block           : %zu\n", args->start_lba);
        fprintf(stderr, "Number of blocks      : %zu\n", 
                DMA_SIZE( args->length, ns_info.lba_data_size ) / ns_info.lba_data_size);
    }

    status = transfer(&ctrl_info, &ns_info, cq, sq, prp_wnd, rw_wnd, transfer_size, args);

    dma_remove(&prp_wnd, &prp_list, args->ctrl_adapter);
    segment_remove(&prp_list);
    dma_remove(&rw_wnd, buffer, args->ctrl_adapter);
    return 0;
}


static int launch(nvm_ctrl_t ctrl, struct cl_args* args, struct segment* buffer)
{
    int status;
    struct segment q_mem;
    nvm_manager_t mngr;
    nvm_rpc_t rpc;
    nvm_dma_t wnd;
    nvm_queue_t cq;
    nvm_queue_t sq;

    status = segment_create(&q_mem, random_id(), ctrl->page_size * 4);
    if (status != 0)
    {
        fprintf(stderr, "Failed to create queue memory: %s\n", strerror(status));
        return 1;
    }
    memset(q_mem.vaddr, 0, q_mem.size);

    // Create DMA window
    status = dma_create(&wnd, ctrl, &q_mem, args->ctrl_adapter);
    if (status != 0)
    {
        segment_remove(&q_mem);
        return 2;
    }

    // Create admin queue manager
    status = nvm_manager_register(&mngr, ctrl, wnd);
    if (status != 0)
    {
        dma_remove(&wnd, &q_mem, args->ctrl_adapter);
        segment_remove(&q_mem);
        fprintf(stderr, "Failed to register manager: %s\n", strerror(status));
        return 2;
    }

    // Get RPC reference to manager
    status = nvm_rpc_bind_local(&rpc, mngr);
    if (status != 0)
    {
        nvm_manager_unregister(mngr);
        dma_remove(&wnd, &q_mem, args->ctrl_adapter);
        segment_remove(&q_mem);
        fprintf(stderr, "Failed to get RPC reference: %s\n", strerror(status));
        return 2;
    }

    // Set number of queues
    status = nvm_rpc_set_num_queues(rpc, 1, 1);
    if (status != 0)
    {
        fprintf(stderr, "Failed to set number of queues: %s\n", strerror(status));
        status = 2;
        goto out;
    }

    // Create queues
    status = nvm_rpc_cq_create(&cq, rpc, ctrl, 1, DMA_VADDR(wnd->vaddr, wnd->page_size, 2), wnd->ioaddrs[2]);
    if (status != 0)
    {
        fprintf(stderr, "Failed to create completion queue: %s\n", strerror(status));
        status = 2;
        goto out;
    }

    status = nvm_rpc_sq_create(&sq, rpc, ctrl, &cq, 1, DMA_VADDR(wnd->vaddr, wnd->page_size, 3), wnd->ioaddrs[3]);
    if (status != 0)
    {
        fprintf(stderr, "Failed to create submission queue: %s\n", strerror(status));
        status = 2;
        goto out;
    }

    status = start_transfer(rpc, ctrl, args, &cq, &sq, buffer);

out:
    nvm_rpc_unbind(rpc);
    nvm_manager_unregister(mngr);
    dma_remove(&wnd, &q_mem, args->ctrl_adapter);
    segment_remove(&q_mem);
    return status;
}


int main(int argc, char** argv)
{
    nvm_ctrl_t ctrl;
    sci_error_t err;

    struct cl_args args;
    parse_opts(argc, argv, &args);

    SCIInitialize(0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to initialize SISCI: %s\n", SCIGetErrorString(err));
        exit(1);
    }

    // Get controller reference
    int status = nvm_dis_ctrl_init(&ctrl, args.smartio_dev_id, args.ctrl_adapter);
    if (status != 0)
    {
        fprintf(stderr, "Failed to get controller reference: %s\n", strerror(status));
        exit(2);
    }

    struct segment buffer;
    status = segment_create(&buffer, random_id(), DMA_SIZE(args.length, ctrl->page_size));
    if (status != 0)
    {
        nvm_ctrl_free(ctrl);
        fprintf(stderr, "Failed to create local buffer\n");
        exit(1);
    }
    
    memset(buffer.vaddr, 0, buffer.size);
    if (args.data != NULL)
    {
        strncpy(((char*) buffer.vaddr) + args.offset, args.data, buffer.size - args.offset);
    }

    status = launch(ctrl, &args, &buffer);

    segment_remove(&buffer);

    // Put controller reference
    nvm_ctrl_free(ctrl);

    SCITerminate();

    if (args.verbose)
    {
        fprintf(stderr, "Goodbye!\n");
    }
    exit(status);
}


static void give_usage(const char* program_name)
{
    fprintf(stderr,
            "Usage: %s --ctrl=<dev id> [--adapter <adapter>] [--write <string>] [--start <blk>] [--length <bytes>]\n",
            program_name);
}


static void show_help(const char* program_name)
{
    give_usage(program_name);
    fprintf(stderr, "    Read or write disk blocks example program.\n\n"
            "    --ctrl         <dev id>    SmartIO device identifier.\n"
            "    --adapter      <adapter>   Local adapter to reach device (default is 0).\n"
            "    --namespace    <ns id>     Namespace identifier (default is 1).\n"
            "    --start        <blk>       Start at given block number (default is 0).\n"
            "    --length       <bytes>     Number of bytes to read/write (default is 1).\n"
            "    --write        <string>    Character string to write.\n"
            "    --offset       <bytes>     Number of bytes to skip into buffer when writing.\n"
            "    --ascii                    Print ASCII characters rather than byte hexadecimal values.\n"
            "    --verbose                  Print more information.\n"
            "    --help                     Show this information.\n"
           "\n");

}


static void parse_opts(int argc, char** argv, struct cl_args* args)
{
    static struct option opts[] = {
        { "help", no_argument, NULL, 'h' },
        { "ctrl", required_argument, NULL, 'c' },
        { "adapter", required_argument, NULL, 'a' },
        { "namespace", required_argument, NULL, 'n' },
        { "start", required_argument, NULL, 's' },
        { "length", required_argument, NULL, 'l' },
        { "offset", required_argument, NULL, 'o' },
        { "write", required_argument, NULL, 'w' },
        { "ascii", no_argument, NULL, 1 },
        { "verbose", no_argument, NULL, 'v' },
        { NULL, 0, NULL, 0 }
    };

    int opt;
    int idx;

    args->data = NULL;
    memset(args, 0, sizeof(struct cl_args));
    
    args->namespace = 1;
    args->length = 1;

    while ((opt = getopt_long(argc, argv, ":hc:a:n:s:l:w:o:v", opts, &idx)) != -1)
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

            case 'v':
                args->verbose = true;
                break;

            case 'h': // show help
                show_help(argv[0]);
                exit(0);
        
            case 'c': // device identifier
                if (parse_u64(optarg, &args->smartio_dev_id, 0) != 0)
                {
                    fprintf(stderr, "Invalid device id: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('c');
                }
                break;

            case 'a': // device adapter
                if (parse_u32(optarg, &args->ctrl_adapter, 10) != 0)
                {
                    fprintf(stderr, "Invalid adapter number: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('a');
                }
                break;

            case 'n': // set namespace
                if (parse_u32(optarg, &args->namespace, 0) != 0 || args->namespace == 0)
                {
                    fprintf(stderr, "Not a valid namespace: %s\n", optarg);
                    exit('n');
                }
                break;

            case 'w': // write instead of read
                args->data = optarg;
                break;

            case 'l': // set length
                if (parse_u64(optarg, &args->length, 0) != 0)
                {
                    fprintf(stderr, "Not a valid number: %s\n", optarg);
                    exit('l');
                }
                break;

            case 's': // set start block
                if (parse_u64(optarg, &args->start_lba, 0) != 0)
                {
                    fprintf(stderr, "Not a valid number: %s\n", optarg);
                    exit('s');
                }
                break;

            case 'o': // set offset
                if (parse_u64(optarg, &args->offset, 0) != 0)
                {
                    fprintf(stderr, "Not a valid number: %s\n", optarg);
                    exit('o');
                }
                break;

            case 1:
                args->use_ascii = true;
                break;
        }
    }

    if (args->offset >= args->length)
    {
        fprintf(stderr, "Invalid offset, can not be larger than length!\n");
        give_usage(argv[0]);
        exit('o');
    }

    if (args->smartio_dev_id == 0)
    {
        fprintf(stderr, "No controller specified!\n");
        give_usage(argv[0]);
        exit('c');
    }

    if (args->length == 0)
    {
        fprintf(stderr, "Length can not be zero!\n");
        exit('l');
    }
}

