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


static int prepare_write(
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

    if (1 < n_prps && n_prps <= 2)
    {
        prp2 = dest->bus_addr[1];
    }
    else if (n_prps > 2)
    {
        prp2 = build_prp_list(page_size, prp_list->virt_addr, n_prps - 1, &prp_list->bus_addr[0], &dest->bus_addr[1]);
    }

    cmd_header(cmd, NVM_WRITE, ns_id);
    cmd_data_ptr(cmd, prp1, prp2);

    cmd->dword[10] = (uint32_t) start_lba;
    cmd->dword[11] = (uint32_t) (start_lba >> 32);

    cmd->dword[12] = (0x00 << 31) | (0x00 < 30) | (0x00 << 26) | n_blks;

    cmd->dword[13] = 0;
    cmd->dword[14] = 0;
    cmd->dword[15] = 0;

    return 0;
}



static int run_workload(nvm_queue_t* cq, nvm_queue_t* sq, buffer_t* prp_list, buffer_t* data, uint32_t ns_id, size_t page_size, uint32_t blk_size)
{
    uint32_t* p = data->virt_addr;
    for (size_t i = 0; i < data->range_size / sizeof(uint32_t); ++i)
    {
        p[i] = 0xcafebabe;
    }

    struct command* cmd = sq_enqueue(sq);
    if (cmd == NULL)
    {
        fprintf(stderr, "Failed to enqueue command\n");
        return EAGAIN;
    }

    prepare_write(cmd, blk_size, page_size, ns_id, 0, data->range_size / blk_size, prp_list, data);

    sq_submit(sq);

    struct completion* cpl = cq_dequeue_block(cq, 5000);
    if (cpl == NULL)
    {
        fprintf(stderr, "Waiting for completion timed out\n");
        return ETIMEDOUT;
    }

    return 0;
}


int workload(nvm_ctrl_t* ctrl, uint32_t ns_id, void* io_mem, size_t io_size)
{
    int err;
    buffer_t* sq_memory = NULL;
    buffer_t* cq_memory = NULL;
    buffer_t* data = NULL;
    buffer_t* prp_list = NULL;
    nvm_queue_t q[2];

    sq_memory = get_buffer(-1, ('s' << 8) | 'q', ctrl->page_size, ctrl->page_size, ctrl->device_id);
    if (sq_memory == NULL)
    {
        err = ENOMEM;
        goto exit;
    }

    cq_memory = get_buffer(-1, ('c' << 8) | 'q', ctrl->page_size, ctrl->page_size, ctrl->device_id);
    if (cq_memory == NULL)
    {
        err = ENOMEM;
        goto exit;
    }

    data = get_buffer(-1, 'd', 0x100000, ctrl->page_size, ctrl->device_id);
    if (data == NULL)
    {
        err = ENOMEM;
        goto exit;
    }

    prp_list = get_buffer(-1, ('p' << 16) | ('r' << 8) | 'p', 0x2000, ctrl->page_size, ctrl->device_id);
    if (prp_list == NULL)
    {
        err = ENOMEM;
        goto exit;
    }


    memset(cq_memory->virt_addr, 0, cq_memory->range_size);
    err = nvm_create_cq(ctrl, &q[0], 1, cq_memory->virt_addr, cq_memory->bus_addr[0], io_mem);
    if (err != 0)
    {
        fprintf(stderr, "Failed to create completion queue: %s\n", strerror(err));
        err = EIO;
        goto exit;
    }

    memset(sq_memory->virt_addr, 0, sq_memory->range_size);
    err = nvm_create_sq(ctrl, &q[0], &q[1], 1, sq_memory->virt_addr, sq_memory->bus_addr[0], io_mem);
    if (err != 0)
    {
        fprintf(stderr, "Failed to create submission queue: %s\n", strerror(err));
        err = EIO;
        goto exit;
    }

    err = run_workload(&q[0], &q[1], prp_list, data, ns_id, ctrl->page_size, 512);

exit:
    put_buffer(data);
    put_buffer(prp_list);
    put_buffer(sq_memory);
    put_buffer(cq_memory);
    return err;
}
