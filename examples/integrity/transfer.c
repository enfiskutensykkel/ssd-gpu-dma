#define _GNU_SOURCE
#include <nvm_types.h>
#include <nvm_util.h>
#include <nvm_cmd.h>
#include <nvm_dma.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_queue.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include "integrity.h"


struct consumer
{
    pthread_t           thread;
    struct queue*       queues;
    uint16_t            n_queues;
    bool                cancel;
};


struct producer
{
    pthread_t           thread;
    bool                write;
    struct buffer*      buffer;
    uint16_t            queue_no;
    struct queue*       queues;
    const struct disk*  disk;
    size_t              start_block;
    size_t              n_blocks;
};


static struct consumer* consume_completions(struct consumer* c)
{
    nvm_cpl_t* cpl;
    nvm_queue_t* cq = &c->queues[0].queue;
    nvm_queue_t* sq = NULL;

    while (!c->cancel)
    {
        cpl = nvm_cq_dequeue_block(cq, 10);

        if (cpl == NULL)
        {
            sched_yield();
            continue;
        }

        sq = &c->queues[*NVM_CPL_SQID(cpl)].queue;
        nvm_sq_update(sq);

        if (!NVM_ERR_OK(cpl))
        {
            fprintf(stderr, "%s\n", nvm_strerror(NVM_ERR_STATUS(cpl)));
        }

        nvm_cq_update(cq);
        c->queues[0].counter++;
    }

    return c;
}


static struct producer* produce_commands(struct producer* p)
{
    nvm_cmd_t* cmd;
    size_t block_size = p->disk->block_size;
    size_t page_size = p->buffer->dma->page_size;

    size_t n_pages = NVM_PAGE_ALIGN(p->n_blocks * block_size, page_size) / page_size; // FIXME: block to page?

    size_t transfer_pages = p->disk->max_data_size / page_size;

    size_t page_base = NVM_BLOCK_TO_PAGE(page_size, block_size, p->start_block);
    size_t page_offset = 0;

    uint32_t ns_id = p->disk->ns_id;

    nvm_dma_t* dma = p->buffer->dma;
    struct queue* queue = &p->queues[p->queue_no];
    nvm_dma_t* prp = queue->qmem.dma;

    nvm_queue_t* sq = &queue->queue;

    while (page_offset < n_pages)
    {
        if (n_pages - page_offset < transfer_pages)
        {
            transfer_pages = n_pages - page_offset;
        }

        while ((cmd = nvm_sq_enqueue(sq)) == NULL)
        {
            nvm_sq_submit(sq);
            sched_yield();
        }

        nvm_cmd_header(cmd, NVM_DEFAULT_CID(sq), p->write ? NVM_IO_WRITE : NVM_IO_READ, ns_id);

        size_t n_blocks = NVM_PAGE_TO_BLOCK(page_size, block_size, transfer_pages);
        size_t start_block = p->start_block + NVM_PAGE_TO_BLOCK(page_size, block_size, page_offset);
        nvm_cmd_rw_blks(cmd, start_block, n_blocks);

        uint16_t prp_no = (*NVM_CMD_CID(cmd) % sq->max_entries) + 1;

        nvm_cmd_data(cmd, page_size, transfer_pages, NVM_DMA_OFFSET(prp, prp_no),
                prp->ioaddrs[prp_no], &dma->ioaddrs[page_base+page_offset]);

        page_offset += transfer_pages;
        queue->counter++;
    }

    if (p->write)
    {
        while ((cmd = nvm_sq_enqueue(sq)) == NULL);
        nvm_cmd_header(cmd, NVM_DEFAULT_CID(sq), NVM_IO_FLUSH, ns_id);
        nvm_cmd_data_ptr(cmd, 0, 0);
        nvm_cmd_rw_blks(cmd, 0, 0);
        queue->counter++;
    }

    nvm_sq_submit(sq);

    return p;
}



static int transfer(const struct disk* d, struct buffer* buffer, struct queue* queues, uint16_t n_queues, off_t size, bool write)
{
    size_t n_blocks = NVM_PAGE_ALIGN(size, d->block_size) / d->block_size;
    size_t n_pages = NVM_PAGE_ALIGN(size, d->page_size) / d->page_size;

    size_t pages_per_queue = n_pages / n_queues;

    struct producer* producers = calloc(n_queues, sizeof(struct producer));
    if (producers == NULL)
    {
        fprintf(stderr, "Failed to allocate thread descriptors\n");
        return -1;
    }

    struct consumer consumer;
    consumer.queues = queues;
    consumer.n_queues = n_queues;
    consumer.cancel = false;

    pthread_create(&consumer.thread, NULL, (void *(*)(void*)) consume_completions, &consumer);

    for (uint16_t i = 0; i < n_queues; ++i)
    {
        producers[i].write = write;
        producers[i].buffer = buffer;
        producers[i].queue_no = i + 1;
        producers[i].queues = queues;
        producers[i].disk = d;
        producers[i].start_block = NVM_PAGE_TO_BLOCK(d->page_size, d->block_size, pages_per_queue * i);
        producers[i].n_blocks = NVM_PAGE_TO_BLOCK(d->page_size, d->block_size, pages_per_queue);

        if (i == n_queues - 1)
        {
            producers[i].n_blocks = n_blocks - producers[i].start_block;
        }

        pthread_create(&producers[i].thread, NULL, (void *(*)(void*)) produce_commands, &producers[i]);
        fprintf(stderr, "\tQueue #%u: block %zu to block %zu (page %zu + %zu)\n", 
                i, producers[i].start_block, producers[i].start_block + producers[i].n_blocks,
                NVM_BLOCK_TO_PAGE(d->page_size, d->block_size, producers[i].start_block),
                NVM_PAGE_ALIGN(producers[i].n_blocks * d->block_size, d->page_size) / d->page_size);

    }

    size_t commands = 0;
    for (uint16_t i = 0; i < n_queues; ++i)
    {
        struct producer* p;
        pthread_join(producers[i].thread, (void**) &p);
        commands += queues[p->queue_no].counter;
    }

    while (queues[0].counter < commands);

    consumer.cancel = true;
    pthread_join(consumer.thread, NULL);
    fprintf(stderr, "Total blocks: %zu\n", n_blocks);


    free(producers);
    return 0;
}



int disk_write(const struct disk* d, struct buffer* buffer, struct queue* queues, uint16_t n_queues, FILE* fp, off_t size)
{
    fread(buffer->dma->vaddr, 1, size, fp);
    return transfer(d, buffer, queues, n_queues, size, true);
}


int disk_read(const struct disk* d, struct buffer* buffer, struct queue* queues, uint16_t n_queues, FILE* fp, off_t size)
{
    int status = transfer(d, buffer, queues, n_queues, size, false);
    if (status == 0)
    {
        fwrite(buffer->dma->vaddr, 1, size, fp);
        fflush(fp);
    }
    return status;
}

