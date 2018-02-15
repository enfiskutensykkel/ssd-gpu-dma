#include <nvm_ctrl.h>
#include <nvm_dma.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "integrity.h"


int create_buffer(struct buffer* b, nvm_aq_ref ref, size_t size, uint32_t adapter, uint32_t id)
{
    int status;

    const nvm_ctrl_t* ctrl = nvm_ctrl_from_aq_ref(ref);

#ifdef __DIS_CLUSTER__
    b->buffer = NULL;
    status = nvm_dis_dma_create(&b->dma, ctrl, adapter, id, size);
#else
    status = posix_memalign(&b->buffer, ctrl->page_size, size);
    if (status != 0)
    {
        fprintf(stderr, "Failed to allocate memory: %s\n", strerror(status));
        return status;
    }

    status = nvm_dma_map_host(&b->dma, ctrl, b->buffer, size);
#endif
    if (!nvm_ok(status))
    {
        free(b->buffer);
        fprintf(stderr, "Failed to create local segment: %s\n", nvm_strerror(status));
        return status;
    }

    memset(b->dma->vaddr, 0, b->dma->page_size * b->dma->n_ioaddrs);

    b->id = id;
    b->adapter = adapter;

    return 0;
}


void remove_buffer(struct buffer* b)
{
    nvm_dma_unmap(b->dma);
    free(b->buffer);
}


int create_queue(struct queue* q, nvm_aq_ref ref, const struct queue* cq, uint16_t qno, uint32_t adapter, uint32_t id)
{
    int status;

    const nvm_ctrl_t* ctrl = nvm_ctrl_from_aq_ref(ref);

    size_t prp_lists = 0;
    if (cq != NULL)
    {
        size_t n_entries = ctrl->page_size / sizeof(nvm_cmd_t);
        prp_lists = n_entries <= ctrl->max_entries ? n_entries : ctrl->max_entries;
    }

    status = create_buffer(&q->qmem, ref, prp_lists * ctrl->page_size + ctrl->page_size, adapter, id);
    if (!nvm_ok(status))
    {
        return status;
    }

    if (cq == NULL)
    {
        status = nvm_admin_cq_create(ref, &q->queue, qno, NVM_DMA_OFFSET(q->qmem.dma, 0), q->qmem.dma->ioaddrs[0]);
    }
    else
    {
        status = nvm_admin_sq_create(ref, &q->queue, &cq->queue, qno, NVM_DMA_OFFSET(q->qmem.dma, 0), q->qmem.dma->ioaddrs[0]);
    }

    if (!nvm_ok(status))
    {
        remove_buffer(&q->qmem);
        fprintf(stderr, "Failed to create queue: %s\n", nvm_strerror(status));
        return status;
    }

    q->counter = 0;
    return 0;
}


void remove_queue(struct queue* q)
{
    remove_buffer(&q->qmem);
}

