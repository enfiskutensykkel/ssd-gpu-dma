#include "nvme.h"
#include "nvme_queue.h"
#include "nvme_core.h"
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>


/* Doorbell register */
#define SQ_DBL(p, y, dstrd)    \
    ((volatile void*) (((volatile unsigned char*) (p)) + /*0x1000 +*/ ((2*(y)) * (4 << (dstrd)))) )


#define CQ_DBL(p, y, dstrd)    \
    ((volatile void*) (((volatile unsigned char*) (p)) + /*0x1000 +*/ ((2*(y) + 1) * (4 << (dstrd)))) )



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
    cq->db = CQ_DBL(controller->dbs, cq->no, controller->dstrd); // FIXME: cq->no ?

    controller->queue_handles[controller->n_queues - 2] = sq;
    controller->queue_handles[controller->n_queues - 1] = cq;

    return 0;
}


int create_queues(nvm_controller_t controller)
{
    return 0;
}
