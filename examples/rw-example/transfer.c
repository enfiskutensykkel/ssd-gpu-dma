#define _GNU_SOURCE
#include <nvm_types.h>
#include <nvm_util.h>
#include <nvm_ctrl.h>
#include <nvm_dma.h>
#include <nvm_queue.h>
#include <nvm_command.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include "transfer.h"


struct dequeue_args
{
    size_t*         remaining_bytes;
    size_t          processed_cpls;
    nvm_queue_t*    cq;
    nvm_queue_t*    sq;
};


static nvm_dma_t create_buffer(nvm_ctrl_t ctrl, size_t size)
{
    void* ptr = NULL;
    int err = posix_memalign(&ptr, ctrl->page_size, size);
    if (err != 0)
    {
        fprintf(stderr, "Failed to allocate page-aligned buffer: %s\n", strerror(err));
        return NULL;
    }

    nvm_dma_t dma_desc;
    err = nvm_dma_window_host_map(&dma_desc, ctrl, ptr, size);
    if (err != 0)
    {
        free(ptr);
        fprintf(stderr, "Failed to map buffer: %s\n", strerror(err));
        return NULL;
    }

    return dma_desc;
}


static void destroy_buffer(nvm_dma_t buffer)
{
    void* ptr = buffer->vaddr;
    nvm_dma_window_free(buffer);
    free(ptr);
}


static size_t dequeue_loop(size_t* remaining, nvm_queue_t* cq, nvm_queue_t* sq)
{
    nvm_cpl_t* cpl;
    size_t processed = 0;

    while (*remaining > 0)
    {
        while ((cpl = cq_dequeue(cq)) != NULL)
        {
            sq_update_unchecked(sq);

            if (!CPL_OK(cpl))
            {
                fprintf(stderr, "Command failed: %s\n", nvm_strerror(cpl));
            }

            ++processed;
        }
    
        cq_update(cq);

        pthread_yield();
    }

    return processed;
}


static void* thread_start(void* args)
{
    struct dequeue_args* da = (struct dequeue_args*) args;
    da->processed_cpls = dequeue_loop(da->remaining_bytes, da->cq, da->sq);
    return da;
}


static int thread_launch(pthread_t* thread_id, size_t* bytes, nvm_queue_t* cq, nvm_queue_t* sq)
{
    struct dequeue_args* args = malloc(sizeof(struct dequeue_args));
    if (args == NULL)
    {
        fprintf(stderr, "Failed to allocate thread arguments\n");
        return errno;
    }

    args->remaining_bytes = bytes;
    args->processed_cpls = 0,
    args->cq = cq;
    args->sq = sq;

    int err = pthread_create(thread_id, NULL, thread_start, args);
    if (err != 0)
    {
        free(args);
        fprintf(stderr, "Failed to start CQ thread: %s\n", strerror(err));
        return err;
    }

    return 0;
}


static int thread_join(pthread_t thread_id, nvm_ctrl_t ctrl, size_t expected_cpls)
{
    struct dequeue_args* args = NULL;

    int err = pthread_join(thread_id, (void**) &args);
    if (err != 0)
    {
        fprintf(stderr, "Failed to join thread: %s\n", strerror(err));
        return err;
    }

    if (args == NULL)
    {
        fprintf(stderr, "This should never happen...\n");
        return 0;
    }

    while (args->processed_cpls < expected_cpls)
    {
        nvm_cpl_t* cpl = cq_dequeue_block(args->cq, ctrl->timeout);
        if (cpl == NULL)
        {
            fprintf(stderr, "WARNING: Controller timed out\n");
            return ETIME;
        }

        sq_update_unchecked(args->sq);
        cq_update(args->cq);

        args->processed_cpls++;
    }

    free(args);
    return 0;
}


static void fill_page_with_same_entry(size_t page_size, void* list_ptr, uint64_t ioaddr)
{
    size_t n_prps = page_size / sizeof(uint64_t);
    uint64_t* entries = (uint64_t*) list_ptr;

    for (size_t i_prp = 0; i_prp < n_prps; ++i_prp)
    {
        entries[i_prp] = ioaddr;
    }
}


static size_t prepare_transfer(nvm_cmd_t* cmd, struct transfer_info* ti, nvm_dma_t buff, size_t* prp_page, size_t* buff_page, size_t remaining)
{
    uint64_t dptr1;
    uint64_t dptr2;
   
    dptr1 = buff->ioaddrs[(*buff_page)++];
    if (remaining <= buff->page_size)
    {
        nvm_cmd_data_ptr(cmd, dptr1, 0);
        return 0;
    }
    else if (remaining <= 2 * buff->page_size)
    {
        dptr2 = buff->ioaddrs[(*buff_page)++];
        nvm_cmd_data_ptr(cmd, dptr1, dptr2);
        return 0;
    }

    size_t chunk_size = _MIN(ti->chunk_size, remaining);
    dptr2 = ti->prp_list->ioaddrs[*prp_page];

    size_t pages_used = nvm_prp_list(DMA_WND_VADDR(ti->prp_list, *prp_page), ti->prp_list->page_size, chunk_size - buff->page_size, &ti->prp_list->ioaddrs[*prp_page], &buff->ioaddrs[*buff_page]);

    *buff_page += pages_used;
    *prp_page += ti->n_prp_pages;

    return remaining - chunk_size;
}


int read_pages(nvm_ctrl_t ctrl, nvm_queue_t* cq, nvm_queue_t* sq, struct transfer_info* ti)
{
    nvm_dma_t dma = create_buffer(ctrl, DMA_SIZE(ti->total_size, ti->blk_size));
    if (dma == NULL)
    {
        return ENOMEM;
    }

    size_t remaining = ti->total_size;
    pthread_t thread;

    int err = thread_launch(&thread, &remaining, cq, sq);
    if (err != 0)
    {
        destroy_buffer(dma);
        return err;
    }

    uint64_t blk_start = ti->start_lba;
    size_t n_cmds = 0;
    size_t prp_page = 0;
    size_t buff_page = 0;

    while (remaining > 0)
    {
        nvm_cmd_t* cmd = sq_enqueue(sq);
        if (cmd == NULL)
        {
            sq_submit(sq);
            pthread_yield();
            continue;
        }

        ++n_cmds;

        size_t chunk_size = _MIN(remaining, ti->chunk_size);
        uint16_t chunk_blocks = DMA_SIZE(chunk_size, ti->blk_size) / ti->blk_size;

        nvm_cmd_header(cmd, NVM_IO_READ, ti->ns);
        remaining = prepare_transfer(cmd, ti, dma, &prp_page, &buff_page, remaining);

        cmd->dword[10] = blk_start;
        cmd->dword[11] = blk_start >> 32;
        cmd->dword[12] = chunk_blocks & 0xffff;

        blk_start += chunk_blocks;
    }

    sq_submit(sq);

    fprintf(stderr, "Waiting for completion...\n");
    err = thread_join(thread, ctrl, n_cmds);
    if (err != 0)
    {
        destroy_buffer(dma);
        return err;
    }

    size_t differs = 0;
    unsigned char* ptr = dma->vaddr;
    for (size_t i = 0; i < dma->page_size * dma->n_ioaddrs; ++i)
    {
        if (*(ptr + i) != 0)
        {
            ++differs;
        }
    }

    destroy_buffer(dma);
    fprintf(stderr, "%zu bytes are non-zero\n", differs);
    return 0;
}


int write_zeros(nvm_ctrl_t ctrl, nvm_queue_t* cq, nvm_queue_t* sq, struct transfer_info* ti)
{
    size_t chunk_size = _MIN(ti->chunk_size, DMA_SIZE(ti->page_size, sizeof(uint64_t)));
    
    nvm_dma_t dma = create_buffer(ctrl, ti->page_size);
    if (dma == NULL)
    {
        return ENOMEM;
    }

    size_t remaining = ti->total_size;

    uint64_t ioaddr = dma->ioaddrs[0];
    fill_page_with_same_entry(ti->prp_list->page_size, ti->prp_list->vaddr, ioaddr);

    pthread_t thread;
    int err = thread_launch(&thread, &remaining, cq, sq);
    if (err != 0)
    {
        destroy_buffer(dma);
        return err;
    }

    uint64_t blk_start = ti->start_lba;
    size_t n_cmds = 0;

    while (remaining > 0)
    {
        nvm_cmd_t* cmd = sq_enqueue(sq);
        if (cmd == NULL)
        {
            sq_submit(sq);
            pthread_yield();
            continue;
        }

        ++n_cmds;
        uint16_t chunk_blks = DMA_SIZE(_MIN(chunk_size, remaining), ti->blk_size) / ti->blk_size;

        nvm_cmd_header(cmd, NVM_IO_WRITE, ti->ns);

        if (remaining <= ti->page_size)
        {
            nvm_cmd_data_ptr(cmd, ioaddr, 0);
            remaining = 0;
        }
        else if (remaining <= 2 * ti->page_size)
        {
            nvm_cmd_data_ptr(cmd, ioaddr, ioaddr);
            remaining = 0;
        }
        else
        {
            nvm_cmd_data_ptr(cmd, ioaddr, ti->prp_list->ioaddrs[0]);
            remaining -= chunk_size;
        }

        cmd->dword[10] = blk_start;
        cmd->dword[11] = blk_start >> 32;
        cmd->dword[12] = chunk_blks & 0xffff;

        blk_start += chunk_blks;

    }

    sq_submit(sq);


    fprintf(stderr, "Waiting for completion...\n");
    err = thread_join(thread, ctrl, n_cmds);
    if (err != 0)
    {
        destroy_buffer(dma);
        return err;
    }

    destroy_buffer(dma);
    fprintf(stderr, "Success!\n");
    return 0;
}

