#include <cuda.h>
#include <nvm_types.h>
#include <nvm_util.h>
#include <nvm_command.h>
#include <nvm_queue.h>
#include <nvm_command.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "benchmark.h"
#include "manager.h"
#include "queue.h"


struct Barrier
{
    const size_t            n_threads;
    size_t                  remaining;
    std::mutex              mtx;
    std::condition_variable cv;

    Barrier(size_t n_threads)
        : n_threads(n_threads)
        , remaining(n_threads)
    {
    }

    void wait()
    {
        std::unique_lock<std::mutex> lock(mtx);

        --remaining;

        if (remaining == 0)
        {
            cv.notify_all();
        }
        else
        {
            cv.wait(lock, [this] {
                return remaining == 0;
            });
        }
    }
};



struct TransferState
{
    size_t      chunk_size;
    size_t      page_size;
    size_t      blk_size;
    size_t      buff_page;
    size_t      prp_page;
    uint64_t    blk_start;
};


static size_t prepare_chunk(TransferTuple& tuple, TransferState& state, DmaPtr prps, DmaPtr buff, size_t remaining)
{
    tuple.page_addr = 0;
    tuple.prp_list_addr = 0;
    tuple.blk_start = 0;
    tuple.n_blks = 0;

    size_t transfer_size = std::min(remaining, state.chunk_size);

    tuple.n_blks = DMA_SIZE(transfer_size, state.blk_size) / state.blk_size;
    tuple.blk_start = state.blk_start;
    state.blk_start += tuple.n_blks;
    tuple.page_addr = (*buff)->ioaddrs[state.buff_page++];

    if (remaining <= state.page_size)
    {
        return 0;
    }
    else if (remaining <= 2 * state.page_size)
    {
        tuple.prp_list_addr = (*buff)->ioaddrs[state.buff_page++];
        return 0;
    }

    void* prp_start = DMA_VADDR((*prps)->vaddr, state.page_size, state.prp_page);
    uint64_t* prp_addr_start = &(*prps)->ioaddrs[state.prp_page];
    uint64_t* buff_addr_start = &(*buff)->ioaddrs[state.buff_page];

    tuple.prp_list_addr = (*buff)->ioaddrs[state.prp_page];

    size_t pages_used = nvm_prp_list(prp_start, state.page_size, transfer_size - state.page_size, prp_addr_start, buff_addr_start);

    state.buff_page += pages_used;
    state.prp_page += nvm_num_prp_pages(state.page_size, transfer_size);

    return remaining - transfer_size;
}


void prepare_transfer_tuples(TupleList& tuples, DmaPtr prps, DmaPtr buff, const BenchmarkSettings& settings)
{
    TransferState state;
    state.chunk_size = settings.chunk_size;
    state.page_size = settings.page_size;
    state.blk_size = settings.blk_size;
    state.buff_page = 0;
    state.prp_page = 0;
    state.blk_start = 0;

    size_t size = settings.total_size;
    while (size != 0)
    {
        TransferTuple tuple;
        size = prepare_chunk(tuple, state, prps, buff, size);
        tuples.push_back(tuple);
    }
}


static uint64_t current_time()
{
    timespec ts;

    if (clock_gettime(CLOCK_REALTIME, &ts) < 0)
    {
        throw std::runtime_error(strerror(errno));
    }

    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}



static void dequeue_loop(Barrier* barrier, QueueManagerPtr queues, size_t total_cmds, size_t* n_cpls, uint64_t timeout)
{
    nvm_cpl_t* cpl;
    size_t i;

    nvm_queue_t* cq = &queues->completion_queue;
    *n_cpls = 0;

    barrier->wait();

    for (i = 0; i < total_cmds; ++i)
    {
        if ((cpl = cq_dequeue_block(cq, timeout)) == NULL)
        {
            return;
        }

        nvm_queue_t* sq = &queues->submission_queues[*CPL_SQID(cpl) - 1];

        sq_update_unchecked(sq);

        if (!CPL_OK(cpl))
        {
            std::cerr << nvm_strerror(cpl) << std::endl;
        }

        cq_update(cq);

        ++*n_cpls;
    }

}


static void enqueue_loop(Barrier* barrier, TupleList::const_iterator start, TupleList::const_iterator end, nvm_queue_t* sq, uint32_t ns, size_t* n_cmds)
{
    nvm_cmd_t* cmd;

    barrier->wait();

    uint32_t fail_count = 0;
    uint32_t tail = 0;

    while (start != end)
    {
        const TransferTuple& transfer = *start;

        cmd = sq_enqueue(sq);
        if (cmd == NULL)
        {
            if (tail != sq->tail)
            {
                tail = sq->tail;
                sq_submit(sq);
            }
            else if (++fail_count == 0x2000)
            {
                return;
            }
            std::this_thread::yield();
            continue;
        }

        fail_count = 0;
        nvm_cmd_header(cmd, NVM_IO_READ, ns);
        nvm_cmd_data_ptr(cmd, transfer.page_addr, transfer.prp_list_addr);
        cmd->dword[10] = transfer.blk_start & ((1ULL << 32) - 1);
        cmd->dword[11] = (transfer.blk_start >> 32);
        cmd->dword[12] = (transfer.n_blks - 1) & 0xffff;

        ++start;
        ++*n_cmds;
    }

    sq_submit(sq);
}



uint64_t time_transfer(TupleList& list, nvm_ctrl_t ctrl, QueueManagerPtr queues, uint32_t ns, void* vaddr, void* devptr, size_t size, int device)
{
    size_t i;
    cudaError_t err;

    err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to set CUDA device");
    }

    Barrier barrier(queues->submission_queues.size() + 2);

    size_t cpls = 0;
    std::thread dequeuer(dequeue_loop, &barrier, queues, list.size(), &cpls, ctrl->timeout);

    std::thread enqueuers[queues->submission_queues.size()];
    size_t cmds[queues->submission_queues.size()];

    auto start = list.cbegin();

    size_t n_trans = list.size() / queues->submission_queues.size();

    for (i = 0; i < queues->submission_queues.size() - 1; ++i)
    {
        cmds[i] = 0;
        enqueuers[i] = std::thread(enqueue_loop, &barrier, start + i * n_trans, start + i * (n_trans+1), &queues->submission_queues[i], ns, &cmds[i]);
    }
    cmds[i] = 0;
    enqueuers[i] = std::thread(enqueue_loop, &barrier, start + i * n_trans, list.cend(), &queues->submission_queues[i], ns, &cmds[i]);

    uint64_t before = current_time();
    barrier.wait();
    
    for (i = 0; i < queues->submission_queues.size(); ++i)
    {
        enqueuers[i].join();
    }
    dequeuer.join();

    size_t total_cmds = 0;
    for (i = 0; i < queues->submission_queues.size(); ++i)
    {
        total_cmds += cmds[i];
    }

    if (total_cmds != list.size())
    {
        throw std::runtime_error("Failed to submit all commands");
    }

    if (cpls != total_cmds)
    {
        throw std::runtime_error("Not all commands completed");
    }

    if (vaddr != nullptr)
    {
        err = cudaMemcpy(devptr, vaddr, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to copy memory from host to device");
        }
    }

    uint64_t after = current_time();
    
    return after - before;
}

