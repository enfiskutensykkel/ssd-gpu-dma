#include <cuda.h>
#include "cuda.h"
#include "nvme.h"
#include "nvme_queue.h"
#include "nvme_core.h"
#include "page.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <stddef.h>
#include <sys/types.h>
#include <unistd.h>


enum
{
    READ            = (0x00 << 7) | (0x00 << 2) | (0x02),
    WRITE           = (0x00 << 7) | (0x00 << 2) | (0x01)
};


__host__ __device__
static int prepare_write(nvm_queue_t sq, uint32_t ns_id, page_t* buf, uint64_t start_lba, uint16_t n_blks)
{
    struct command* cmd = sq_enqueue(sq);
    if (cmd == NULL)
    {
        return EAGAIN;
    }

    // Set command header
    cmd->dword[0] |= (0x00 << 14) | (0x00 << 8) | WRITE;

    // Specify namespace
    cmd->dword[1] = ns_id;

    cmd->dword[4] = 0;
    cmd->dword[5] = 0;

    uint64_t phys_addr = buf->phys_addr;
    cmd->dword[6] = (uint32_t) phys_addr;
    cmd->dword[7] = (uint32_t) (phys_addr >> 32);
    cmd->dword[8] = 0;
    cmd->dword[9] = 0;

    cmd->dword[10] = (uint32_t) start_lba;
    cmd->dword[11] = (uint32_t) (start_lba >> 32);

    cmd->dword[12] = n_blks;
    cmd->dword[13] = 0;
    cmd->dword[14] = 0;
    cmd->dword[15] = 0;

    return 0;
}


__host__ __device__
static int prepare_read(nvm_queue_t sq, uint32_t ns_id, page_t* buf, uint64_t start_lba, uint16_t n_blks)
{
    struct command* cmd = sq_enqueue(sq);
    if (cmd == NULL)
    {
        return EAGAIN;
    }

    // Set command header
    cmd->dword[0] |= (0x00 << 14) | (0x00 << 8) | READ;

    // Specify namespace
    cmd->dword[1] = ns_id;

    // Zero out metadata pointer
    cmd->dword[4] = 0;
    cmd->dword[5] = 0;

    // Set data transfer pointer
    uint64_t phys_addr = buf->phys_addr;
    cmd->dword[6] = (uint32_t) phys_addr;
    cmd->dword[7] = (uint32_t) (phys_addr >> 32);
    cmd->dword[8] = 0;
    cmd->dword[9] = 0;

    // Command dword 10 and 11 (starting LBA)
    cmd->dword[10] = (uint32_t) start_lba;
    cmd->dword[11] = (uint32_t) (start_lba >> 32);

    // Command dword 12
    cmd->dword[12] = (0x00 << 31) | (0x00 << 30) | (0x00 << 26) | n_blks;

    // Command dword 13-15
    cmd->dword[13] = 0;
    cmd->dword[14] = 0;
    cmd->dword[15] = 0;

    return 0;
}


static int create_io_queues(nvm_controller_t ctrl, int fd, int dev, size_t num_pairs)
{
    int status;

    // Prepare a number of IO queue pairs
    for (size_t i = 0; i < num_pairs; ++i)
    {
        status = prepare_queue_handles(ctrl);
        if (status != 0)
        {
            fprintf(stderr, "Failed to create queue handles\n");
            return status;
        }


        nvm_queue_t sq = ctrl->queue_handles[ctrl->n_queues - 2];
        status = get_page(&sq->page, fd, dev);
        if (status != 0)
        {
            fprintf(stderr, "Failed to allocate and pin SQ memory\n");
            return status;
        }

        memset(sq->page.virt_addr, 0, sq->page.page_size);

        nvm_queue_t cq = ctrl->queue_handles[ctrl->n_queues - 1];
        status = get_page(&cq->page, fd, dev);
        if (status != 0)
        {
            fprintf(stderr, "Failed to allocate and pin CQ memory\n");
            return status;
        }

        memset(cq->page.virt_addr, 0, cq->page.page_size);
    }

    // Submit all prepared queues to the controller
    status = create_queues(ctrl);
    if (status != 0)
    {
        fprintf(stderr, "Failed to submit prepared queues\n");
        return status;
    }

    return 0;
}


int start_kernel(int fd, int dev, nvm_controller_t ctrl)
{
    int status;
    page_t data;

    status = get_page(&data, fd, 0);
    if (status != 0)
    {
        fprintf(stderr, "Failed to create data buffer\n");
        return status;
    }

//    printf("value in mem=%x\n", *((volatile unsigned int*) data.virt_addr));
//    volatile uint32_t* x = (volatile uint32_t*) data.virt_addr;
//
    status = create_io_queues(ctrl, fd, dev, 1);
    if (status != 0)
    {
        fprintf(stderr, "Failed to create IO queues\n");
        return status;
    }

//*x = 0xdeadbeef;
//prepare_write(ctrl->queue_handles[2], 1, &data, 0, 1);

    prepare_read(ctrl->queue_handles[2], 1, &data, 0, 1);

    printf("submitting\n");
    sq_submit(ctrl->queue_handles[2]);

    printf("waiting for completion\n");
    struct completion* cpl;
    while ((cpl = cq_dequeue(ctrl->queue_handles[3], ctrl)) == NULL);
    printf("%x\n", STATUS_CODE(cpl));

    cq_update(ctrl->queue_handles[3]);

    usleep(500000);
//    printf("value in mem=%x\n", *((volatile unsigned int*) data.virt_addr));
        
    uint32_t x = 0;

    cudaMemcpy((void*) &x, data.virt_addr, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("value: %x\n", x);

    return 0;
}
