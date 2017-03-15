#include "nvme.h"
#include "nvme_queue.h"
#include "nvme_core.h"
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>


/* Some admin command opcodes */
enum
{
    DELETE_IO_SUBMISSION_QUEUE          = (0x00 << 7) | (0x00 << 2) | 0x00,
    CREATE_IO_SUBMISSION_QUEUE          = (0x00 << 7) | (0x00 << 2) | 0x01,
    DELETE_IO_COMPLETION_QUEUE          = (0x00 << 7) | (0x01 << 2) | 0x00,
    CREATE_IO_COMPLETION_QUEUE          = (0x00 << 7) | (0x01 << 2) | 0x01,
    ABORT                               = (0x00 << 7) | (0x02 << 2) | 0x00
};


/* SQ doorbell register */
#define SQ_DBL(p, y, dstrd)    \
    ((volatile void*) (((volatile unsigned char*) (p)) + 0x1000 + ((2*(y)) * (4 << (dstrd)))) )


/* CQ doorbell register */
#define CQ_DBL(p, y, dstrd)    \
    ((volatile void*) (((volatile unsigned char*) (p)) + 0x1000 + ((2*(y) + 1) * (4 << (dstrd)))) )



static void clear_page(page_t* page)
{
    page->device = -1;
    page->kernel_handle = -1;
    page->virt_addr = NULL;
    page->phys_addr = (uint64_t) NULL;
    page->page_size = 0;
}


static void clear_queue_handle(nvm_queue_t queue, nvm_controller_t controller, uint16_t no)
{
    queue->no = no;
    clear_page(&queue->page);
    queue->max_entries = controller->max_entries;
    queue->entry_size = (no % 2 == 0) ? controller->sq_entry_size : controller->cq_entry_size;
    queue->head = 0;
    queue->tail = 0;
    queue->phase = 1;
    queue->db = NULL;
}


int prepare_queue_handles(nvm_controller_t controller)
{
    if (controller->n_queues >= (controller->max_queues - 1))
    {
        fprintf(stderr, "Maximum number of queues already created\n");
        return ENOSPC;
    }

    struct nvm_queue* cq = malloc(sizeof(struct nvm_queue));
    if (cq == NULL)
    {
        fprintf(stderr, "Failed to allocate queue handle: %s\n", strerror(errno));
        return errno;
    }

    struct nvm_queue* sq = malloc(sizeof(struct nvm_queue));
    if (sq == NULL)
    {
        fprintf(stderr, "Failed to allocate queue handle: %s\n", strerror(errno));
        free(cq);
        return errno;
    }

    controller->n_queues += 2;
    clear_queue_handle(sq, controller, controller->n_queues - 2);
    clear_queue_handle(cq, controller, controller->n_queues - 1);

    sq->db = SQ_DBL(controller->dbs, sq->no, controller->dstrd);
    cq->db = CQ_DBL(controller->dbs, cq->no, controller->dstrd); // FIXME: Should this really be cq->no ?

    controller->queue_handles[controller->n_queues - 2] = sq;
    controller->queue_handles[controller->n_queues - 1] = cq;

    return 0;
}


static int create_cq(nvm_controller_t controller, nvm_queue_t queue)
{
    struct command* cmd = sq_enqueue(controller->queue_handles[0]);
    if (cmd == NULL)
    {
        return -EAGAIN;
    }

    cmd->dword[0] |= (0x00 << 14) | (0x00 << 8) | CREATE_IO_COMPLETION_QUEUE;
    cmd->dword[1] = 0;

    // Set metadata pointer
    cmd->dword[4] = 0;
    cmd->dword[5] = 0;

    // Set data pointer
    uint64_t data_ptr = queue->page.phys_addr;
    cmd->dword[6] = (uint32_t) data_ptr;
    cmd->dword[7] = (uint32_t) (data_ptr >> 32);
    cmd->dword[8] = 0;
    cmd->dword[9] = 0;

    cmd->dword[10] = ((queue->max_entries  - 1) << 16) | (queue->no / 2 + 1);

    // Interrupt vector | Interrupts enable | Physically contiguous 
    cmd->dword[11] = (0x0000 << 16) | (0x00 << 1) | 0x01;

    // Submit command and wait for completion
    sq_submit(controller->queue_handles[0]);

    struct completion* cpl;
    while ((cpl = cq_dequeue(controller->queue_handles[1], controller)) == NULL);

    int status = STATUS_CODE(cpl);

    cq_update(controller->queue_handles[1]);

    return status;
}


static int create_sq(nvm_controller_t controller, nvm_queue_t queue)
{
    struct command* cmd = sq_enqueue(controller->queue_handles[0]);
    if (cmd == NULL)
    {
        return -EAGAIN;
    }

    cmd->dword[0] |= (0x00 << 14) | (0x00 << 8) | CREATE_IO_SUBMISSION_QUEUE;
    cmd->dword[1] = 0;

    // Set metadata pointer
    cmd->dword[4] = 0;
    cmd->dword[5] = 0;

    // Set data pointer
    uint64_t data_ptr = queue->page.phys_addr;
    cmd->dword[6] = (uint32_t) data_ptr;
    cmd->dword[7] = (uint32_t) (data_ptr >> 32);
    cmd->dword[8] = 0;
    cmd->dword[9] = 0;

    cmd->dword[10] = ((queue->max_entries  - 1) << 16) | queue->no;

    // Completion queue id | queue priority | PC */
    cmd->dword[11] = ((queue->no / 2 + 1) << 16) | (0x00 << 1) | 0x01;

    // Submit command and wait for completion
    sq_submit(controller->queue_handles[0]);

    struct completion* cpl;
    while ((cpl = cq_dequeue(controller->queue_handles[1], controller)) == NULL);

    int status = STATUS_CODE(cpl);

    cq_update(controller->queue_handles[1]);

    return status;
}


int create_queues(nvm_controller_t controller)
{
    int status;

    /* Create IO queue pairs */
    for (uint16_t i = 2; i < controller->n_queues; i += 2)
    {
        status = create_cq(controller, controller->queue_handles[i + 1]);
        if (status != 0)
        {
            fprintf(stderr, "An error occured while creating IO queues\n");
            if (status < 0)
            {
                return -status;
            }
            return EIO;
        }

        status = create_sq(controller, controller->queue_handles[i]);
        if (status != 0)
        {
            fprintf(stderr, "An error occured while creating IO queues\n");
            if (status < 0)
            {
                return -status;
            }
            return EIO;
        }
    }

    return 0;
}
