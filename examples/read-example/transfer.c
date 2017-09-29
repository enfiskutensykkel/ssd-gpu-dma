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


static int create_buffer(nvm_dma_t* buffer, nvm_ctrl_t ctrl, size_t size)
{
    int err;
    void* ptr = NULL;
    *buffer = NULL;

    err = posix_memalign(&ptr, ctrl->page_size, size);
    if (err != 0)
    {
        fprintf(stderr, " Failed to allocate page-aligned buffer: %s\n", strerror(err));
        return err;
    }

    err = nvm_dma_window_host_map(buffer, ctrl, ptr, size);
    if (err != 0)
    {
        free(ptr);
        *buffer = NULL;
        fprintf(stderr, "Failed to map buffer: %s\n", strerror(err));
        return err;
    }

    return 0;
}


static void destroy_buffer(nvm_dma_t buffer)
{
    if (buffer != NULL)
    {
        void* ptr = buffer->vaddr;
        nvm_dma_window_free(buffer);
        free(ptr);
    }
}


struct thread_args
{
    int*            flag;
    nvm_queue_t*    cq;
    nvm_queue_t*    sq;
    size_t          cpls;
};


static void* process_completions(struct thread_args* args)
{
    nvm_cpl_t* cpl = NULL;
    nvm_queue_t* cq = args->cq;
    nvm_queue_t* sq = args->sq;
    size_t cpls = 0;

    while (*args->flag)
    {
        while ((cpl = cq_dequeue(cq)) != NULL)
        {
            sq_update_unchecked(sq);

            if (!CPL_OK(cpl))
            {
                fprintf(stderr, "Command failed: %s\n", nvm_strerror(cpl));
            }

            ++cpls;
        }

        cq_update(cq);
        pthread_yield();
    }

    args->cpls = cpls;
    return args;
}


static int completion_thread_start(pthread_t* thread, int* flag, nvm_queue_t* cq, nvm_queue_t* sq)
{
    struct thread_args* args = malloc(sizeof(struct thread_args));
    if (args == NULL)
    {
        fprintf(stderr, "Failed to allocate CQ thread data: %s\n", strerror(errno));
        return ENOMEM;
    }

    args->flag = flag;
    args->cq = cq;
    args->sq = sq;
    args->cpls = 0;

    int err = pthread_create(thread, NULL, (void* (*)(void*)) process_completions, args);
    if (err != 0)
    {
        free(args);
        fprintf(stderr, "Failed to create CQ thread: %s\n", strerror(err));
        return err;
    }

    return 0;
}


static int completion_thread_join(pthread_t thread, uint64_t timeout, size_t expected)
{
    int err;
    struct thread_args* args = NULL;

    err = pthread_join(thread, (void**) &args);
    if (err != 0)
    {
        fprintf(stderr, "Failed to join CQ thread: %s\n", strerror(err));
        return err;
    }

    if (args == NULL)
    {
        fprintf(stderr, "This should not happen...\n");
        return 0;
    }

    while (args->cpls < expected)
    {
        nvm_cpl_t* cpl = cq_dequeue_block(args->cq, timeout);
        if (cpl == NULL)
        {
            fprintf(stderr, "Controller timed out!\n");
            return ETIME;
        }

        sq_update_unchecked(args->sq);

        if (!CPL_OK(cpl))
        {
            fprintf(stderr, "Command failed: %s\n", nvm_strerror(cpl));
        }

        cq_update(args->cq);
        args->cpls++;
    }

    free(args);
    return 0;
}


int read_pages(nvm_ctrl_t ctrl, nvm_queue_t* cq, nvm_queue_t* sq, const struct transfer_info* ti)
{
    int err;
    int run = 1;
    nvm_dma_t buffer = NULL;
    nvm_dma_t prp_list = NULL;
    size_t cmds = 0;
    size_t nonzero = 0;
    volatile unsigned char* ptr;

    err = create_buffer(&buffer, ctrl, ti->blk_size * ti->n_blks);
    if (err != 0)
    {
        goto out;
    }
    memset(buffer->vaddr, 0xff, ti->blk_size * ti->n_blks);

    size_t chunk_size = DMA_ALIGN(ti->chunk_size, ti->blk_size);
    size_t n_prp_pages = nvm_num_prp_pages(ctrl->page_size, chunk_size);
    size_t n_prp_lists = (ti->blk_size * ti->n_blks) / chunk_size 
        + ((ti->blk_size * ti->n_blks) % chunk_size >= 2 * ctrl->page_size);

    err = create_buffer(&prp_list, ctrl, n_prp_lists * n_prp_pages * ctrl->page_size);
    if (err != 0)
    {
        goto out;
    }

    pthread_t thread;
    err = completion_thread_start(&thread, &run, cq, sq);
    if (err != 0)
    {
        goto out;
    }

    size_t remaining_blks = ti->n_blks;
    size_t buffer_page = 0;
    uint64_t current_blk = ti->start_lba;
    size_t current_list = 0;

    fprintf(stderr, "Reading from disk...\n");
    while (remaining_blks > 0)
    {
        size_t transfer_blks = _MIN(remaining_blks, chunk_size / ti->blk_size);
        size_t transfer_size = transfer_blks * ti->blk_size;

        nvm_cmd_t* cmd = sq_enqueue(sq);
        if (cmd == NULL)
        {
            sq_submit(sq);
            pthread_yield();
            continue;
        }

        ++cmds;
        uint64_t dptr1 = buffer->ioaddrs[buffer_page++];
        uint64_t dptr2 = 0;

        if (transfer_size <= buffer->page_size)
        {
            dptr2 = 0;
        }
        else if (transfer_size <= 2 * buffer->page_size)
        {
            dptr2 = buffer->ioaddrs[buffer_page++];
        }
        else
        {
            dptr2 = prp_list->ioaddrs[current_list];
            
            size_t prp_list_offset = current_list++ * n_prp_pages;

            buffer_page += nvm_prp_list(DMA_WND_VADDR(prp_list, prp_list_offset), buffer->page_size, 
                    transfer_size - buffer->page_size, &prp_list->ioaddrs[prp_list_offset], &buffer->ioaddrs[buffer_page]);
        }

        nvm_cmd_header(cmd, NVM_IO_READ, ti->ns);
        nvm_cmd_data_ptr(cmd, dptr1, dptr2);

        cmd->dword[10] = current_blk;
        cmd->dword[11] = current_blk >> 32;
        cmd->dword[12] = (transfer_blks - 1) & 0xffff;
        cmd->dword[13] = 0;
        cmd->dword[14] = 0;
        cmd->dword[15] = 0;

        current_blk += transfer_blks;
        remaining_blks -= transfer_blks;
    }

    sq_submit(sq);
    pthread_yield();

    fprintf(stderr, "Waiting for completions...\n");
    run = 0;
    completion_thread_join(thread, ctrl->timeout, cmds);
    
out:
    ptr = buffer->vaddr;
    for (size_t i = 0, n = ti->n_blks * ti->blk_size; i < n; ++i)
    {
        if (ptr[i] != 0)
        {
            nonzero += 1;
        }
    }

    destroy_buffer(prp_list);
    destroy_buffer(buffer);
    fprintf(stderr, "Commands used: %zu, number of non-zero bytes: %zx\n", cmds, nonzero);
    return 0;
}


int write_zeros(nvm_ctrl_t ctrl, nvm_queue_t* cq, nvm_queue_t* sq, const struct transfer_info* ti)
{
    nvm_cmd_t* cmd;
    size_t remaining_blks = ti->n_blks;
    uint64_t blk_offset = ti->start_lba;
    size_t cmds = 0;
    int run = 1;
    int err;

    nvm_dma_t buffer;
    err = create_buffer(&buffer, ctrl, ctrl->page_size);
    if (err != 0)
    {
        return err;
    }
    memset(buffer->vaddr, 0x00, ctrl->page_size);

    pthread_t thread;
    err = completion_thread_start(&thread, &run, cq, sq);
    if (err != 0)
    {
        destroy_buffer(buffer);
        return err;
    }

    fprintf(stderr, "Writing zeroes to disk...\n");
    while (remaining_blks > 0)
    {
        //size_t current_blks = _MIN(remaining_blks, ti->chunk_size / ti->blk_size);
        size_t current_blks = _MIN(remaining_blks, DMA_SIZE(2 * ctrl->page_size, ti->blk_size) / ti->blk_size);

        cmd = sq_enqueue(sq);
        if (cmd == NULL)
        {
            sq_submit(sq);
            pthread_yield();
            continue;
        }
        ++cmds;

        //nvm_cmd_header(cmd, NVM_IO_WRITE_ZEROES, ti->ns);
        //nvm_cmd_data_ptr(cmd, 0, 0);

        nvm_cmd_header(cmd, NVM_IO_WRITE, ti->ns);
        nvm_cmd_data_ptr(cmd, buffer->ioaddrs[0], buffer->ioaddrs[0]);

        cmd->dword[10] = blk_offset;
        cmd->dword[11] = blk_offset >> 32;
        cmd->dword[12] = (current_blks - 1) & 0xffff;
        cmd->dword[14] = 0;
        cmd->dword[15] = 0;

        remaining_blks -= current_blks;
        blk_offset += current_blks;
    }

    sq_submit(sq);
    pthread_yield();

    fprintf(stderr, "Waiting for completion...\n");
    run = 0;
    completion_thread_join(thread, ctrl->timeout, cmds);

    destroy_buffer(buffer);
    return 0;
}

