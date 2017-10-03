#include <cuda.h>
#include <nvm_types.h>
#include <nvm_command.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <algorithm>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <ctime>
#include "settings.h"
#include "dma.h"
#include "transfer.h"
#include "benchmark.h"


struct Barrier
{
    size_t numRemaining;
    std::condition_variable cv;
    std::mutex mtx;

    Barrier(size_t n)
        : numRemaining(n)
    { }

    void wait()
    {
        std::unique_lock<std::mutex> lock(mtx);

        if (--numRemaining == 0)
        {
            cv.notify_all();
        }
        else
        {
            cv.wait(lock, [this] { return numRemaining == 0; });
        }

        lock.unlock();
    }
};


static uint64_t currentTime()
{
    timespec ts;

    if (clock_gettime(CLOCK_REALTIME, &ts) < 0)
    {
        throw std::runtime_error(strerror(errno));
    }

    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}


static void dequeueCompletions(nvm_queue_t** queues, size_t totalCommands)
{
    nvm_cpl_t* cpl = nullptr;
    nvm_queue_t* cq = queues[0];
    nvm_queue_t* sq;
    size_t cplCount = 0;

    while (cplCount < totalCommands)
    {
        while ((cpl = cq_dequeue(cq)) != nullptr)
        {
            sq = queues[*CPL_SQID(cpl)];
            sq_update_unchecked(sq);

            if (!CPL_OK(cpl))
            {
                fprintf(stderr, "Command failed: %s\n", nvm_strerror(cpl));
            }

            ++cplCount;
        }

        cq_update(cq);
    }
}


static void enqueueCommands(Barrier* barrier, const TransferPtr transfer)
{
    nvm_cmd_t* cmd = nullptr;
    uint32_t ns = transfer->nvmNamespace;
    nvm_queue_t* sq = transfer->queue;

    barrier->wait();

    for (const auto& chunk: transfer->chunks)
    {
        while ((cmd = sq_enqueue(sq)) == nullptr)
        {
            sq_submit(sq);
            continue;
        }

        nvm_cmd_header(cmd, NVM_IO_READ, ns);
        nvm_cmd_dptr(cmd, chunk.pageIoAddr, chunk.prpListIoAddr);

        cmd->dword[10] = chunk.startBlock;
        cmd->dword[11] = chunk.startBlock >> 32;
        cmd->dword[12] = (chunk.numBlocks - 1) & 0xffff;
    }

    sq_submit(sq);
}


static uint64_t timeTransfer(QueueList& queueList, const TransferList& transfers, size_t totalCommands)
{
    // Create queue array
    nvm_queue_t* queues[queueList.size()];
    for (size_t i = 0; i < queueList.size(); ++i)
    {
        queues[i] = &queueList[i];
    }

    // Create completion dequeuer thread
    std::thread dequeuer(dequeueCompletions, (nvm_queue_t**) queues, totalCommands);

    // Create thread barrier to synchronize all threads
    Barrier barrier(transfers.size() + 1);

    // Create command enqueuer threads
    std::thread enqueuers[transfers.size()];
    for (const auto& transfer: transfers)
    {
        enqueuers[transfer->queue->no - 1] = std::thread(enqueueCommands, &barrier, transfer);
    }
    
    uint64_t before = currentTime();
    barrier.wait();

    // Wait for all threads to complete
    for (size_t i = 0; i < transfers.size(); ++i)
    {
        enqueuers[i].join();
    }

    uint64_t after = currentTime();

    dequeuer.join();
    
    return after - before;
}


double benchmark(QueueList& queueList, const TransferList& transfers, DmaPtr hostBuffer, DmaPtr deviceBuffer)
{
    // Count total number of commands and transfer size
    size_t totalCommands = 0;
    size_t transferSize = 0;

    std::for_each(transfers.begin(), transfers.end(), [&transferSize,&totalCommands](const TransferPtr& transfer) {
        totalCommands += transfer->chunks.size();
        
        for (const auto& chunk: transfer->chunks)
        {
            transferSize += chunk.numBlocks * transfer->blockSize;
        }
    });

    uint64_t transferTime = timeTransfer(queueList, transfers, totalCommands);

    // Copy from host memory to device
    uint64_t before = currentTime();
    cudaError_t err = cudaMemcpy((*deviceBuffer)->vaddr, (*hostBuffer)->vaddr, transferSize, cudaMemcpyHostToDevice);
    uint64_t after = currentTime();
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to copy to device");
    }

    transferTime += after - before;

    return transferSize / (double) transferTime;
}


double benchmark(QueueList& queueList, const TransferList& transfers, DmaPtr deviceBuffer)
{
    // Count total number of commands and transfer size
    size_t totalCommands = 0;
    size_t transferSize = 0;

    std::for_each(transfers.begin(), transfers.end(), [&transferSize,&totalCommands](const TransferPtr& transfer) {
        totalCommands += transfer->chunks.size();
        
        for (const auto& chunk: transfer->chunks)
        {
            transferSize += chunk.numBlocks * transfer->blockSize;
        }
    });

    uint64_t transferTime = timeTransfer(queueList, transfers, totalCommands);

    return transferSize / (double) transferTime;
}

