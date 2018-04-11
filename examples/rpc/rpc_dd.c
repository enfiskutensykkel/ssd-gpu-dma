#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_admin.h>
#include <nvm_dma.h>
#include <nvm_rpc.h>
#include <nvm_aq.h>
#include <nvm_util.h>
#include <nvm_error.h>
#include <nvm_queue.h>
#include <nvm_cmd.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <getopt.h>
#include <string.h>
#include <errno.h>
#include <sisci_api.h>
#include <signal.h>
#include <pthread.h>
#include <unistd.h>
#include "segment.h"
#include "util.h"

#define MIN(a, b)   ( (a) <= (b) ? (a) : (b) )


struct arguments
{
    uint32_t    segment_id;
    uint32_t    adapter;
    uint16_t    queue_id;
    uint64_t    controller_id;
    uint32_t    nvm_namespace;
    uint64_t    offset;
    uint64_t    blocks;
    uint64_t    count;
    FILE*       input;
    bool        server;
    uint16_t    num_queues;
};


struct disk_info
{
    uint32_t    ns_id;
    size_t      page_size;
    size_t      chunk_size;
    size_t      block_size;
    size_t      max_blocks;
};


struct queue_pair
{
    size_t      cmds;
    size_t      cpls;
    size_t      errors;
    nvm_queue_t cq;
    nvm_queue_t sq;
};


static volatile bool signal_caught = false;


static void catch_signal()
{
    signal_caught = true;
    fprintf(stderr, "Stopping!\n");
}



static void give_usage(const char* progname)
{
    fprintf(stderr,
            "Usage: %s --ctrl <fdid> --qn <number> [--offset <count>] --count <count>\n"
            "   or: %s --ctrl <fdid> --server --nq <number> --qn <number> [--offset <count>] --count <count>\n"
            "   or: %s --ctrl <fdid> --qn <number> [--offset <count>] --input\n"
            "   or: %s --ctrl <fdid> --server --nq <number> --qn <number> [--offset <count>] --input\n"
            , progname, progname, progname, progname);
}


static void parse_args(int argc, char** argv, struct arguments* args);



static void disk_w(const struct disk_info* disk, struct queue_pair* qp, FILE* fp, uint64_t blks, uint64_t offset, nvm_dma_t* wnd)
{
    void*           buf_ptr     = NVM_DMA_OFFSET(wnd, 3);
    const uint64_t* buf_ioaddrs = &wnd->ioaddrs[3];

    void*           prp_ptr     = NVM_DMA_OFFSET(wnd, 2);
    const uint64_t  prp_ioaddr  = wnd->ioaddrs[2];

    while (!signal_caught && !feof(fp) && !ferror(fp))
    {
        size_t is = fread(buf_ptr, 1, blks * disk->block_size, fp);
        size_t curr_blks = NVM_PAGE_ALIGN(is, disk->block_size) / disk->block_size;

        nvm_cmd_t* cmd;
        while (!signal_caught && (cmd = nvm_sq_enqueue(&qp->sq)) == NULL)
        {
            usleep(1);
        }

        if (signal_caught)
        {
            break;
        }

        nvm_cmd_header(cmd, NVM_DEFAULT_CID(&qp->sq), NVM_IO_WRITE, disk->ns_id);
        nvm_cmd_data(cmd, disk->page_size, NVM_PAGE_ALIGN(curr_blks * disk->block_size, disk->page_size),
                    prp_ptr, prp_ioaddr, buf_ioaddrs);
        nvm_cmd_rw_blks(cmd, offset, curr_blks);

        nvm_cache_flush(buf_ptr, NVM_PAGE_ALIGN(curr_blks * disk->block_size, disk->page_size));

        nvm_sq_submit(&qp->sq);
        qp->cmds++;

        nvm_cpl_t* cpl;
        while (!signal_caught && (cpl = nvm_cq_dequeue_block(&qp->cq, 50)) == NULL);

        if (signal_caught)
        {
            break;
        }

        if (cpl == NULL)
        {
            fprintf(stderr, "Missing completion!\n");
            qp->errors++;
            continue;
        }

        nvm_sq_update(&qp->sq);

        if (!NVM_ERR_OK(cpl))
        {
            fprintf(stderr, "Agh! %s\n", nvm_strerror(NVM_ERR_STATUS(cpl)));
            qp->errors++;
        }

        nvm_cq_update(&qp->cq);
        qp->cpls++;

        offset += curr_blks;
    }

    nvm_cmd_t* cmd = nvm_sq_enqueue(&qp->sq);
    if (cmd == NULL)
    {
        qp->errors++;
        return;
    }

    nvm_cmd_header(cmd, NVM_DEFAULT_CID(&qp->sq), NVM_IO_FLUSH, disk->ns_id);
    nvm_cmd_data_ptr(cmd, 0, 0);
    nvm_sq_submit(&qp->sq);
}


static void disk_r(const struct disk_info* disk, struct queue_pair* qp, uint64_t blks, uint64_t count, uint64_t offset, nvm_dma_t* wnd)
{
    void*           buf_ptr     = NVM_DMA_OFFSET(wnd, 3);
    const uint64_t* buf_ioaddrs = &wnd->ioaddrs[3];

    void*           prp_ptr     = NVM_DMA_OFFSET(wnd, 2);
    const uint64_t  prp_ioaddr  = wnd->ioaddrs[2];

    while (!signal_caught && count != 0)
    {
        size_t curr_blks = MIN(blks, count);

        nvm_cmd_t* cmd;
        while (!signal_caught && (cmd = nvm_sq_enqueue(&qp->sq)) == NULL)
        {
            usleep(1);
        }

        nvm_cmd_header(cmd, NVM_DEFAULT_CID(&qp->sq), NVM_IO_READ, disk->ns_id);
        nvm_cmd_data(cmd, disk->page_size, NVM_PAGE_ALIGN(curr_blks * disk->block_size, disk->page_size),
                    prp_ptr, prp_ioaddr, buf_ioaddrs);
        nvm_cmd_rw_blks(cmd, offset, curr_blks);

        nvm_cache_flush(buf_ptr, NVM_PAGE_ALIGN(curr_blks * disk->block_size, disk->page_size));

        nvm_sq_submit(&qp->sq);
        qp->cmds++;

        nvm_cpl_t* cpl;
        while (!signal_caught && (cpl = nvm_cq_dequeue_block(&qp->cq, 50)) == NULL);

        if (signal_caught)
        {
            break;
        }

        if (cpl == NULL)
        {
            fprintf(stderr, "Missing completion!\n");
            qp->errors++;
            continue;
        }

        nvm_sq_update(&qp->sq);

        if (!NVM_ERR_OK(cpl))
        {
            fprintf(stderr, "Agh! %s\n", nvm_strerror(NVM_ERR_STATUS(cpl)));
            qp->errors++;
        }

        nvm_cq_update(&qp->cq);
        qp->cpls++;

        nvm_cache_invalidate(buf_ptr, NVM_PAGE_ALIGN(curr_blks * disk->block_size, disk->page_size));

        fwrite(buf_ptr, 1, curr_blks * disk->block_size, stdout);

        offset += curr_blks;
        count -= curr_blks;
    }
}


static int disk_rw(const struct disk_info* disk, struct queue_pair* qp, const struct arguments* args, nvm_dma_t* window)
{
    int status = 0;
    qp->cmds = 0;
    qp->cpls = 0;
    qp->errors = 0;

    if (args->input != NULL)
    {
        fprintf(stderr, "buffer size=%zu blks\n", args->blocks);
        disk_w(disk, qp, args->input, args->blocks, args->offset, window);
    }
    else
    {
        fprintf(stderr, "buffer size=%zu blks, count=%zu\n", args->blocks, args->count);
        disk_r(disk, qp, args->blocks, args->count, args->offset, window);
    }

    if (qp->cmds != qp->cpls)
    {
        fprintf(stderr, "Agh! cmds=%zu cpls=%zu\n", qp->cmds, qp->cpls);
        status = EPIPE;
    }

    if (qp->errors > 0)
    {
        fprintf(stderr, "Owf!! errors: %zu\n", qp->errors);
        status = EIO;
    }

    return status;
}


static int get_disk_info(nvm_aq_ref rpc, uint32_t ns_id, uint32_t seg_id, uint32_t adapter, struct disk_info* disk)
{
    struct segment page;
    nvm_dma_t* dma;

    int status = segment_create(&page, seg_id, 0x1000);
    if (status != 0)
    {
        fprintf(stderr, "Failed to info segment\n");
        return status;
    }

    status = dma_create(&dma, nvm_ctrl_from_aq_ref(rpc), &page, adapter);
    if (status != 0)
    {
        segment_remove(&page);
        fprintf(stderr, "Failed to create DMA window\n");
        return status;
    }

    struct nvm_ctrl_info ctrl_info;
    status = nvm_admin_ctrl_info(rpc, &ctrl_info, dma->vaddr, dma->ioaddrs[0]);
    if (!nvm_ok(status))
    {
        fprintf(stderr, "Failed to identify controller: %s\n", nvm_strerror(status));
        goto leave;
    }

    struct nvm_ns_info ns_info;
    status = nvm_admin_ns_info(rpc, &ns_info, ns_id, dma->vaddr, dma->ioaddrs[0]);
    if (!nvm_ok(status))
    {
        fprintf(stderr, "Failed to identify namespace: %s\n", nvm_strerror(status));
        goto leave;
    }

    disk->ns_id = ns_info.ns_id;
    disk->page_size = ctrl_info.page_size;
    disk->chunk_size = ctrl_info.max_data_size;
    disk->block_size = ns_info.lba_data_size;
    disk->max_blocks = ns_info.size;

leave:
    dma_remove(dma, &page, adapter);
    segment_remove(&page);
    return status;
}


static void destroy_queues(nvm_aq_ref rpc, struct queue_pair* qp)
{
    int err = nvm_admin_sq_delete(rpc, &qp->sq, &qp->cq);
    if (!nvm_ok(err))
    {
        fprintf(stderr, "Failed to delete SQ: %s\n", nvm_strerror(err));
    }

    err = nvm_admin_cq_delete(rpc, &qp->cq);
    if (!nvm_ok(err))
    {
        fprintf(stderr, "Failed to delete CQ: %s\n", nvm_strerror(err));
    }
}


static int create_queues(nvm_aq_ref rpc, uint16_t queue_id, nvm_dma_t* dma, struct queue_pair* qp)
{
    int status;

    status = nvm_admin_cq_create(rpc, &qp->cq, queue_id, NVM_DMA_OFFSET(dma, 0), dma->ioaddrs[0]);
    if (!nvm_ok(status))
    {
        fprintf(stderr, "Failed to create IO completion queue (CQ): %s\n", nvm_strerror(status));
        return status;
    }

    status = nvm_admin_sq_create(rpc, &qp->sq, &qp->cq, queue_id, NVM_DMA_OFFSET(dma, 1), dma->ioaddrs[1]);
    if (!nvm_ok(status))
    {
        fprintf(stderr, "Failed to create IO submission queue (SQ): %s\n", nvm_strerror(status));
        return status;
    }

    return 0;
}


static int prepare_buffer(nvm_aq_ref rpc, 
                          const struct arguments* args, 
                          const struct disk_info* disk, 
                          struct segment* segment,
                          nvm_dma_t** dma)
{
    const nvm_ctrl_t* ctrl = nvm_ctrl_from_aq_ref(rpc);
    size_t size = 3 * disk->page_size + NVM_CTRL_ALIGN(ctrl, disk->block_size * args->blocks);

    int status = segment_create(segment, args->segment_id, size);
    if (status != 0)
    {
        fprintf(stderr, "Failed to create buffer segment\n");
        return status;
    }

    status = dma_create(dma, ctrl, segment, args->adapter);
    if (status != 0)
    {
        segment_remove(segment);
        fprintf(stderr, "Failed to create DMA window\n");
        return status;
    }

    memset((*dma)->vaddr, 0, size);
    nvm_cache_flush((*dma)->vaddr,  size);

    return status;
}



int main(int argc, char** argv)
{
    nvm_ctrl_t* ctrl;
    nvm_aq_ref rpc;
    struct segment segment;
    nvm_dma_t* dma_window;
    struct segment aq_mem;
    nvm_dma_t* aq_window;

    struct arguments args;
    parse_args(argc, argv, &args);

    sci_error_t err;
    SCIInitialize(0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to initialize SISCI: %s\n", SCIGetErrorString(err));
        exit(1);
    }

    signal(SIGTERM, (sig_t) catch_signal);
    signal(SIGINT, (sig_t) catch_signal);
    signal(SIGPIPE, (sig_t) catch_signal);

    // Get controller reference
    int status = nvm_dis_ctrl_init(&ctrl, args.controller_id, args.adapter);
    if (status != 0)
    {
        fprintf(stderr, "Failed to get controller reference: %s\n", strerror(status));
        exit(2);
    }

    if (args.server)
    {
        // Become RPC server
        status = segment_create(&aq_mem, args.segment_id + 1, 2 * ctrl->page_size);
        if (status != 0)
        {
            nvm_ctrl_free(ctrl);
            SCITerminate();
            fprintf(stderr, "Failed to become RPC server: %s\n", strerror(status));
            exit(2);
        }

        status = dma_create(&aq_window, ctrl, &aq_mem, args.adapter);
        if (status != 0)
        {
            segment_remove(&aq_mem);
            nvm_ctrl_free(ctrl);
            SCITerminate();
            fprintf(stderr, "Failed to become RPC server: %s\n", strerror(status));
            exit(2);
        }

        memset(aq_window->vaddr, 0, 2 * aq_window->page_size);

        status = nvm_aq_create(&rpc, ctrl, aq_window);
        if (status != 0)
        {
            dma_remove(aq_window, &aq_mem, args.adapter);
            segment_remove(&aq_mem);
            nvm_ctrl_free(ctrl);
            SCITerminate();
            fprintf(stderr, "Failed to become RPC server: %s\n", strerror(status));
            exit(2);
        }

        status = nvm_dis_rpc_enable(rpc, args.adapter, NULL);
        if (status != 0)
        {
            fprintf(stderr, "Ouch! %s\n", nvm_strerror(status));
            goto leave;
        }

        uint16_t n_cqs = args.num_queues;
        uint16_t n_sqs = args.num_queues;
        status = nvm_admin_request_num_queues(rpc, &n_cqs, &n_sqs);
        if (!nvm_ok(status))
        {
            fprintf(stderr, "Ack!! %s\n", nvm_strerror(status));
            goto leave;
        }

        if (n_sqs < args.num_queues || n_cqs < args.num_queues)
        {
            fprintf(stderr, "Failed to reserve SQ/CQ queue-pairs\n");
            goto leave;
        }
    }
    else
    {
        // Get RPC reference
        status = nvm_dis_rpc_bind(&rpc, ctrl, args.adapter);
        if (status != 0)
        {
            nvm_ctrl_free(ctrl);
            SCITerminate();
            fprintf(stderr, "Failed to get RPC reference: %s\n", strerror(status));
            exit(2);
        }
    }

    // Get information about the disk
    struct disk_info disk;
    status = get_disk_info(rpc, args.nvm_namespace, args.segment_id, args.adapter, &disk);
    if (status != 0)
    {
        goto leave;
    }

    if (args.blocks == 0)
    {
        args.blocks = disk.chunk_size / disk.block_size;
    }

    // Check if buffer size is sensible
    if (disk.chunk_size < args.blocks * disk.block_size)
    {
        fprintf(stderr, "Buffer/chunk size exceeds controller's maximum data transfer size\n");
        status = 3;
        goto leave;
    }

    // Check if read size is sensible
    if (args.offset + args.count > disk.max_blocks)
    {
        fprintf(stderr, "Trying to read larger than disk size\n");
        status = 3;
        goto leave;
    }

    // Create buffer and set up IO queues
    status = prepare_buffer(rpc, &args, &disk, &segment, &dma_window);
    if (status != 0)
    {
        goto leave;
    }

    struct queue_pair queues;
    status = create_queues(rpc, args.queue_id, dma_window, &queues);
    if (status != 0)
    {
        goto remove;
    }

    // Do the cool stuff
    status = disk_rw(&disk, &queues, &args, dma_window);

    destroy_queues(rpc, &queues);

remove:
    dma_remove(dma_window, &segment, args.adapter);
    segment_remove(&segment);

    if (args.server)
    {
        nvm_dis_rpc_disable(rpc, args.adapter);
    }

leave:
    //nvm_rpc_unbind(rpc);
    nvm_aq_destroy(rpc);

    if (args.server)
    {
        dma_remove(aq_window, &aq_mem, args.adapter);
        segment_remove(&aq_mem);
    }

    nvm_ctrl_free(ctrl);
    SCITerminate();
    fprintf(stderr, "Exit status: %s\n", strerror(status));
    exit(status);
}


static void show_help(const char* progname)
{
    give_usage(progname);
    fprintf(stderr,
            "\nRead or write blocks from disk using a remote controller manager in a DIS cluster.\n\n"
            "    --ctrl         <fdid>      SmartIO device identifier (fabric device id).\n"
            "    --adapter      <adapter>   Local adapter to reach device (default is 0).\n"
            "    --namespace    <ns id>     NVM namespace identifier (default is 1).\n"
            "    --ns           <ns id>     Short-hand for --namespace.\n"
            "    --segment-id   <seg id>    Local segment identifier (default is random).\n"
            "    --queue-id     <number>    SQ/CQ queue pair number.\n"
            "    --queue-num    <number>    Alternative to --queue-id.\n"
            "    --qn           <number>    Short-hand for --queue-id.\n"
            "    --offset       <count>     Number of blocks to offset (default is 0).\n"
            "    --buffer       <count>     Buffer size in number of blocks (default is max).\n"
            "    --count        <count>     Total number of blocks to read.\n"
            "    --blocks       <count>     Alternative to --count.\n"
            "    --input                    Read from stdin and write to disk.\n"
            "    --num-queues   <number>    Number of queue-pairs to reserve.\n"
            "    --nq           <number>    Short-hand for --num-queues.\n"
            "    --server                   Act as RPC server.\n"
            "    --help                     Show this information.\n"
            "\n");
}


static void parse_args(int argc, char** argv, struct arguments* args)
{
    static struct option opts[] = {
        {"help", no_argument, NULL, 'h'},
        {"ctrl", required_argument, NULL, 'd'},
        {"adapter", required_argument, NULL, 'a'},
        {"namespace", required_argument, NULL, 1},
        {"nvm-ns", required_argument, NULL, 1},
        {"ns", required_argument, NULL, 1},
        {"segment-id", required_argument, NULL, 0},
        {"queue-id", required_argument, NULL, 'q'},
        {"queue-num", required_argument, NULL, 'q'},
        {"qn", required_argument, NULL, 'q'},
        {"offset", required_argument, NULL, 'o'},
        {"buffer", required_argument, NULL, 'b'},
        {"blocks", required_argument, NULL, 'c'},
        {"count", required_argument, NULL, 'c'},
        {"num-queues", required_argument, NULL, 'n'},
        {"nq", required_argument, NULL, 'n'},
        {"server", no_argument, NULL, 's'},
        {"input", no_argument, NULL, 'i'},
        {NULL, 0, NULL, 0}
    };

    args->segment_id = random_id();
    args->adapter = 0;
    args->queue_id = 0;
    args->controller_id = 0;
    args->nvm_namespace = 1;
    args->offset = 0;
    args->blocks = 0;
    args->count = 0;
    args->input = NULL;
    args->server = false;
    args->num_queues = 0;

    int opt;
    int idx;

    while ((opt = getopt_long(argc, argv, ":hd:a:n:sq:o:b:c:i", opts, &idx)) != -1)
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

            case 'd': // device identifier
                if (parse_u64(optarg, &args->controller_id, 16) != 0)
                {
                    fprintf(stderr, "Invalid device id: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('c');
                }
                break;

            case 'a': // device adapter
                if (parse_u32(optarg, &args->adapter, 10) != 0)
                {
                    fprintf(stderr, "Invalid adapter number: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('a');
                }
                break;

            case 1: // NVM namespace
                if (parse_u32(optarg, &args->nvm_namespace, 0) != 0 || args->nvm_namespace == 0)
                {
                    fprintf(stderr, "Invalid NVM namespace: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('n');
                }
                break;

            case 0: // Segment identifier
                if (parse_u32(optarg, &args->segment_id, 0) != 0)
                {
                    fprintf(stderr, "Invalid segment id: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('s');
                }
                break;

            case 'q': // Queue number
                if (parse_u16(optarg, &args->queue_id, 0) != 0 || args->queue_id == 0)
                {
                    fprintf(stderr, "Invalid SQ/CQ queue-pair id: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('q');
                }
                break;

            case 'o': // Block offset
                if (parse_u64(optarg, &args->offset, 0) != 0)
                {
                    fprintf(stderr, "Invalid block count: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('b');
                }
                break;

            case 'b': // Buffer size
                if (parse_u64(optarg, &args->blocks, 0) != 0 || args->blocks == 0)
                {
                    fprintf(stderr, "Invalid block count: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('b');
                }
                break;

            case 'c': // Loop count
                if (parse_u64(optarg, &args->count, 0) != 0 || args->count == 0)
                {
                    fprintf(stderr, "Invalid count: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('b');
                }
                break;

            case 'n': // Number of queues
                if (parse_u16(optarg, &args->num_queues, 0) != 0 || args->num_queues == 0)
                {
                    fprintf(stderr, "Invalid number of SQ/CQ queue-pairs: %s\n", optarg);
                    give_usage(argv[0]);
                    exit('q');
                }
                break;

            case 's':
                args->server = true;
                break;

            case 'i':
                args->input = stdin;
                break;
        }
    }

    if (args->controller_id == 0)
    {
        fprintf(stderr, "No controller specified!\n");
        give_usage(argv[0]);
        exit('c');
    }

    if (args->input == NULL && args->count == 0)
    {
        fprintf(stderr, "Either count or input must be specified\n");
        give_usage(argv[0]);
        exit('c');
    }
    else if (args->input != NULL && args->count != 0)
    {
        fprintf(stderr, "Only one of count or input can be specified\n");
        give_usage(argv[0]);
        exit('i');
    }

    if (args->queue_id == 0)
    {
        fprintf(stderr, "Must specify SQ/CQ queue-pair id\n");
        give_usage(argv[0]);
        exit('q');
    }

    if (args->server && args->num_queues == 0)
    {
        fprintf(stderr, "Option --server requires option --num-queues (or --nq)\n");
        give_usage(argv[0]);
        exit('n');
    }

    if (args->server && args->num_queues < args->queue_id)
    {
        fprintf(stderr, "SQ/CQ queue-pair id must be lower than total number of queue-pairs\n");
        give_usage(argv[0]);
        exit('n');
    } 
}
