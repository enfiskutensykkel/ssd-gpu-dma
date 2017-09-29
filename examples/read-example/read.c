#define _GNU_SOURCE
#include <nvm_types.h>
#include <nvm_util.h>
#include <nvm_ctrl.h>
#include <nvm_dma.h>
#include <nvm_manager.h>
#include <nvm_rpc.h>
#include <nvm_queue.h>
#include <nvm_command.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <pthread.h>
#include <unistd.h>
#include "transfer.h"
#include "util.h"


static int load_transfer_info(struct transfer_info* ti, nvm_rpc_t ref, nvm_ctrl_t ctrl)
{
    int err;
    size_t page_size = _MAX(ctrl->page_size, 0x1000);
    void* page_ptr = NULL;
    nvm_dma_t page_dma = NULL;

    err = posix_memalign(&page_ptr, ctrl->page_size, page_size);
    if (err != 0)
    {
        fprintf(stderr, "Failed to allocate page-sized buffer: %s\n", strerror(err));
        goto out;
    }

    err = nvm_dma_window_host_map(&page_dma, ctrl, page_ptr, page_size);
    if (err != 0)
    {
        fprintf(stderr, "Failed to map page buffer: %s\n", strerror(err));
        goto out;
    }

    nvm_ctrl_info_t ci;
    err = nvm_rpc_ctrl_info(&ci, ref, ctrl, page_dma->vaddr, page_dma->ioaddrs[0]);
    if (err != 0)
    {
        fprintf(stderr, "Failed to get controller information: %s\n", strerror(err));
        goto out;
    }

    nvm_ns_info_t ni;
    err = nvm_rpc_ns_info(&ni, ref, ti->ns, page_dma->vaddr, page_dma->ioaddrs[0]);
    if (err != 0)
    {
        fprintf(stderr, "Failed to get namespace information: %s\n", strerror(err));
        goto out;
    }

    ti->page_size = ctrl->page_size;
    ti->blk_size = ni.lba_data_size;
    ti->chunk_size = ci.max_transfer_size;

out:
    nvm_dma_window_free(page_dma);
    free(page_ptr);
    return err;
}


static int create_queues(nvm_ctrl_t ctrl, nvm_rpc_t ref, nvm_dma_t qdma, nvm_queue_t* cq, nvm_queue_t* sq)
{
    int err;

    err = nvm_rpc_set_num_queues(ref, 1, 1);
    if (err != 0)
    {
        fprintf(stderr, "Failed to set number of queues: %s\n", strerror(err));
        return 1;
    }

    err = nvm_rpc_cq_create(cq, ref, ctrl, 1, DMA_VADDR(qdma->vaddr, qdma->page_size, 2), qdma->ioaddrs[2]);
    if (err != 0)
    {
        fprintf(stderr, "Failed to create CQ: %s\n", strerror(err));
        return 1;
    }

    err = nvm_rpc_sq_create(sq, ref, ctrl, cq, 1, DMA_VADDR(qdma->vaddr, qdma->page_size, 3), qdma->ioaddrs[3]);
    if (err != 0)
    {
        fprintf(stderr, "Failed to create SQ: %s\n", strerror(err));
        return err;
    }

    return 0;
}


static void parse_args(int argc, char** argv, struct transfer_info* ti, uint64_t* ctrl_id, bool* write);


int main(int argc, char** argv)
{
    struct transfer_info ti;
    uint64_t ctrl_id = 0;
    bool write = false;
    parse_args(argc, argv, &ti, &ctrl_id, &write);

    nvm_ctrl_t ctrl;
    int err = nvm_ctrl_init(&ctrl, ctrl_id);
    if (err != 0)
    {
        fprintf(stderr, "Failed to get controller reference: %s\n", strerror(err));
        exit(1);
    }

    void* aq_mem;
    err = posix_memalign(&aq_mem, ctrl->page_size, ctrl->page_size * 4);
    if (err != 0)
    {
        fprintf(stderr, "Failed to allocate page-aligned memory for admin queues: %s\n", strerror(errno));
        nvm_ctrl_free(ctrl);
        exit(2);
    }

    nvm_dma_t aq_dma;
    err = nvm_dma_window_host_map(&aq_dma, ctrl, aq_mem, ctrl->page_size * 4);
    if (err != 0)
    {
        free(aq_mem);
        nvm_ctrl_free(ctrl);
        fprintf(stderr, "Failed to create DMA window mapping: %s\n", strerror(errno));
        exit(1);
    }
    memset(aq_mem, 0, ctrl->page_size * 4);

    fprintf(stderr, "Resetting controller....\n");
    nvm_manager_t mngr;
    err = nvm_manager_register(&mngr, ctrl, aq_dma);
    if (err != 0)
    {
        nvm_dma_window_free(aq_dma);
        free(aq_mem);
        nvm_ctrl_free(ctrl);
        fprintf(stderr, "Failed to initialize controller: %s\n", strerror(err));
        exit(1);
    }

    nvm_rpc_t ref;
    nvm_rpc_bind_local(&ref, mngr);

    nvm_queue_t cq;
    nvm_queue_t sq;
    err = create_queues(ctrl, ref, aq_dma, &cq, &sq);
    if (err != 0)
    {
        goto out;
    }

    err = load_transfer_info(&ti, ref, ctrl);
    if (err != 0)
    {
        goto out;
    }

    if (write)
    {
        err = write_zeros(ctrl, &cq, &sq, &ti);
        if (err != 0)
        {
            goto out;
        }
    }
    
    err = read_pages(ctrl, &cq, &sq, &ti);

out:
    nvm_rpc_unbind(ref);
    nvm_manager_unregister(mngr);
    nvm_dma_window_free(aq_dma);
    free(aq_mem);
    nvm_ctrl_free(ctrl);
    exit(err);
}


static void give_usage(const char* name)
{
    fprintf(stderr, "Usage: %s --ctrl=<ctrl id> --namespace=<ns id> --blocks=<num> [--start=<block>] [--zero]\n", name);
}


static void show_help(const char* name)
{
    give_usage(name);
    fprintf(stderr, "    Read blocks from disk.\n\n"
            "    --ctrl         <ctrl id>   Device ID ('/dev/disnvmeXXX'). Default is 0.\n"
            "    --namespace    <ns id>     Set namespace (default is 1).\n"
            "    --blocks       <num>       Number of blocks (default is 1).\n"
            "    --start        <block>     Start block (default is 0).\n"
            "    --zero                     Write 0s first and read back.\n"
            "    --help                     Show this information.\n");
}



static void parse_args(int argc, char** argv, struct transfer_info* ti, uint64_t* ctrl_id, bool* write)
{
    static struct option opts[] = {
        { "help", no_argument, NULL, 'h' },
        { "ctrl", required_argument, NULL, 'c' },
        { "namespace", required_argument, NULL, 'n' },
        { "blocks", required_argument, NULL, 'b' },
        { "start", required_argument, NULL, 's' },
        { "zero", no_argument, NULL, 'z' },
        { NULL, 0, NULL, 0 }
    };

    int opt;
    int idx;
    
    memset(ti, 0, sizeof(struct transfer_info));
    ti->ns = 1;
    ti->n_blks = 1;

    while ((opt = getopt_long(argc, argv, ":hc:n:b:s:z", opts, &idx)) != -1)
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
                if (parse_u64(optarg, ctrl_id, 10) != 0)
                {
                    fprintf(stderr, "Invalid controller identifier: `%s'\n", optarg);
                    give_usage(argv[0]);
                    exit('c');
                }
                break;

            case 'n': // set namespace
                if (parse_u32(optarg, &ti->ns, 0) != 0 || ti->ns == 0)
                {
                    fprintf(stderr, "Not a valid namespace: `%s'\n", optarg);
                    exit('n');
                }
                break;

            case 'b': // set number of blocks
                if (parse_u64(optarg, &ti->n_blks, 0) != 0 || ti->n_blks == 0)
                {
                    fprintf(stderr, "Invalid number of blocks: `%s'\n", optarg);
                    exit('s');
                }
                break;

            case 's': // set start block
                if (parse_u64(optarg, &ti->start_lba, 0) != 0)
                {
                    fprintf(stderr, "Not a valid number: `%s'\n", optarg);
                    exit('s');
                }
                break;

            case 'z':
                *write = true;
                break;

            case 'h':
                show_help(argv[0]);
                exit(0);
        }
    }
}
