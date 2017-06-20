#include "memory.h"
#include "nvm/types.h"
#include "nvm/queue.h"
#include "nvm/command.h"
#include "nvm/util.h"
#include "nvm/ctrl.h"
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


static int prepare_io(
        enum nvm_command_set id,
        struct command* cmd, 
        uint32_t blk_size, 
        uint32_t page_size, 
        uint32_t ns_id, 
        uint64_t start_lba, 
        uint16_t n_blks, 
        buffer_t* prp_list, 
        buffer_t* dest
        )
{
    size_t blks_per_page = page_size / blk_size;
    size_t n_prps = n_blks / blks_per_page;

    if (n_blks < 1)
    {
        return ERANGE;
    }

    uint64_t prp1 = dest->bus_addr[0];
    uint64_t prp2 = 0;

    if (n_prps > 1)
    {
        prp2 = build_prp_list(page_size, prp_list->virt_addr, n_prps - 1, &prp_list->bus_addr[0], &dest->bus_addr[1]);
    }

    fprintf(stderr, "ns_id=%u start_lba=%lu n_blks=%u\n", ns_id, start_lba, n_blks);
    fprintf(stderr, "prp1=%lx prp2=%lx\n", prp1, prp2);

    cmd_header(cmd, id, ns_id);
    cmd_data_ptr(cmd, prp1, prp2);

    cmd->dword[10] = (uint32_t) start_lba;
    cmd->dword[11] = (uint32_t) (start_lba >> 32);

    cmd->dword[12] = (0x00 << 31) | (0x00 < 30) | (0x00 << 26) | (n_blks - 1);

    cmd->dword[13] = 0;
    cmd->dword[14] = 0;
    cmd->dword[15] = 0;

    return 0;
}


static int run_workload(nvm_queue_t* cq, nvm_queue_t* sq, buffer_t* prp_list, buffer_t* data, uint32_t ns_id, size_t page_size, uint32_t blk_size)
{
    uint16_t* p = data->virt_addr;
    p[0] = rand() % 0xff;
    for (size_t i = 1; i < data->range_size / sizeof(uint16_t); ++i)
    {
        p[i] = p[i - 1] + 1;
    }

    struct command* cmd = sq_enqueue(sq);
    prepare_io(NVM_WRITE, cmd, blk_size, page_size, ns_id, 0, data->range_size / blk_size, prp_list, data);
    sq_submit(sq);

    cmd = sq_enqueue(sq);
    prepare_io(NVM_READ, cmd, blk_size, page_size, ns_id, 0, data->range_size / blk_size, prp_list, data);

    struct completion* cpl = cq_dequeue_block(cq, 5000);
    if (cpl == NULL)
    {
        fprintf(stderr, "Waiting for completion timed out\n");
        return ETIME;
    }
    sq_update(sq, cpl);

    printf("cid=%u sqhd=%u sqid=%u sct=%u sc=%u\n", 
            *CPL_CID(cpl), *CPL_SQHD(cpl), *CPL_SQID(cpl), SCT(cpl), SC(cpl));

    memset(data->virt_addr, 0, data->range_size);

    sq_submit(sq);
    cpl = cq_dequeue_block(cq, 5000);
    if (cpl == NULL)
    {
        fprintf(stderr, "Waiting for completion timed out\n");
        return ETIME;
    }
    sq_update(sq, cpl);
    cq_update(cq);

    printf("cid=%u sqhd=%u sqid=%u sct=%u sc=%u\n", 
            *CPL_CID(cpl), *CPL_SQHD(cpl), *CPL_SQID(cpl), SCT(cpl), SC(cpl));

    size_t corrects = 0;
    size_t faults = 0;
    for (size_t i = 1; i < data->range_size / sizeof(uint16_t); ++i)
    {
        corrects += (p[i] - 1 == p[i - 1]);
        faults += (p[i] - 1 != p[i - 1]);
    }

    fprintf(stdout, "corrects=%zu faults=%zu total=%zu\n", corrects, faults, data->range_size / sizeof(uint16_t));
    return 0;
}


int workload(sci_device_t device, uint32_t node_id, uint32_t intno, uint32_t ns_id, void* io_mem, size_t io_size)
{
    size_t ctrl_page_size = 0x1000;

    int err;
    buffer_t* sq_memory = NULL;
    buffer_t* cq_memory = NULL;
    buffer_t* data = NULL;
    buffer_t* prp_list = NULL;
    nvm_queue_t q[2];
    size_t prps_per_page = ctrl_page_size / sizeof(uint64_t);
    size_t prp_list_size;
    size_t data_size = _MIN(40 * 512 /* FIXME: hack */, 0x1000 * 512);

    srand(time(NULL));

    sq_memory = get_buffer(-1, ('s' << 8) | 'q', ctrl_page_size, ctrl_page_size, device);
    if (sq_memory == NULL)
    {
        err = ENOMEM;
        goto exit;
    }

    cq_memory = get_buffer(-1, ('c' << 8) | 'q', ctrl_page_size, ctrl_page_size, device);
    if (cq_memory == NULL)
    {
        err = ENOMEM;
        goto exit;
    }

    data = get_buffer(-1, 'd', data_size, ctrl_page_size, device);
    if (data == NULL)
    {
        fprintf(stderr, "Failed to allocate data buffer\n");
        err = ENOMEM;
        goto exit;
    }

    prp_list_size = data->range_size / ctrl_page_size / prps_per_page + 1;

    prp_list = get_buffer(-1, ('p' << 16) | ('r' << 8) | 'p', prp_list_size * ctrl_page_size, ctrl_page_size, device);
    if (prp_list == NULL)
    {
        err = ENOMEM;
        goto exit;
    }

    memset(cq_memory->virt_addr, 0, cq_memory->range_size);
    err = nvm_create_cq(node_id, intno, &q[0], 1, cq_memory->virt_addr, cq_memory->bus_addr[0], io_mem);
    if (err != 0)
    {
        fprintf(stderr, "Failed to create completion queue: %s\n", strerror(err));
        err = EIO;
        goto exit;
    }

    memset(sq_memory->virt_addr, 0, sq_memory->range_size);
    err = nvm_create_sq(node_id, intno, &q[0], &q[1], 1, sq_memory->virt_addr, sq_memory->bus_addr[0], io_mem);
    if (err != 0)
    {
        fprintf(stderr, "Failed to create submission queue: %s\n", strerror(err));
        err = EIO;
        goto exit;
    }

    err = run_workload(&q[0], &q[1], prp_list, data, ns_id, ctrl_page_size, 512);

exit:
    put_buffer(data);
    put_buffer(prp_list);
    put_buffer(sq_memory);
    put_buffer(cq_memory);
    return err;
}

//int workload(nvm_ctrl_t* ctrl, uint32_t ns_id, void* io_mem, size_t io_size)
//{
//    int err;
//    buffer_t* sq_memory = NULL;
//    buffer_t* cq_memory = NULL;
//    buffer_t* data = NULL;
//    buffer_t* prp_list = NULL;
//    nvm_queue_t q[2];
//    size_t prps_per_page = ctrl->page_size / sizeof(uint64_t);
//    size_t prp_list_size;
//    size_t data_size = _MIN(ctrl->max_data_size, 0x1000 * 512);
//
//    srand(time(NULL));
//
//    sq_memory = get_buffer(-1, ('s' << 8) | 'q', ctrl->page_size, ctrl->page_size, ctrl->device);
//    if (sq_memory == NULL)
//    {
//        err = ENOMEM;
//        goto exit;
//    }
//
//    cq_memory = get_buffer(-1, ('c' << 8) | 'q', ctrl->page_size, ctrl->page_size, ctrl->device);
//    if (cq_memory == NULL)
//    {
//        err = ENOMEM;
//        goto exit;
//    }
//
//    fprintf(stdout, "MDTS = %zu\n", ctrl->max_data_size);
//    data = get_buffer(-1, 'd', data_size, ctrl->page_size, ctrl->device);
//    if (data == NULL)
//    {
//        fprintf(stderr, "Failed to allocate data buffer\n");
//        err = ENOMEM;
//        goto exit;
//    }
//
//    prp_list_size = data->range_size / ctrl->page_size / prps_per_page + 1;
//    fprintf(stdout, "PRP list size = %zu\n", prp_list_size);
//
//    prp_list = get_buffer(-1, ('p' << 16) | ('r' << 8) | 'p', prp_list_size * ctrl->page_size, ctrl->page_size, ctrl->device);
//    if (prp_list == NULL)
//    {
//        err = ENOMEM;
//        goto exit;
//    }
//
//    memset(cq_memory->virt_addr, 0, cq_memory->range_size);
//    err = nvm_create_cq(ctrl, &q[0], 1, cq_memory->virt_addr, cq_memory->bus_addr[0], io_mem);
//    if (err != 0)
//    {
//        fprintf(stderr, "Failed to create completion queue: %s\n", strerror(err));
//        err = EIO;
//        goto exit;
//    }
//
//    memset(sq_memory->virt_addr, 0, sq_memory->range_size);
//    err = nvm_create_sq(ctrl, &q[0], &q[1], 1, sq_memory->virt_addr, sq_memory->bus_addr[0], io_mem);
//    if (err != 0)
//    {
//        fprintf(stderr, "Failed to create submission queue: %s\n", strerror(err));
//        err = EIO;
//        goto exit;
//    }
//
//    err = run_workload(&q[0], &q[1], prp_list, data, ns_id, ctrl->page_size, 512);
//
//exit:
//    put_buffer(data);
//    put_buffer(prp_list);
//    put_buffer(sq_memory);
//    put_buffer(cq_memory);
//    return err;
//}
