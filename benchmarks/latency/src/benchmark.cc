#include <memory>
#include <map>
#include <vector>
#include <chrono>
#include <thread>
#include <functional>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "benchmark.h"
#include "ctrl.h"
#include "buffer.h"
#include "transfer.h"
#include "settings.h"
#include "barrier.h"
#include <nvm_types.h>
#include <nvm_util.h>
#include <nvm_error.h>
#include <nvm_queue.h>
#include <nvm_cmd.h>

using std::thread;
using error = std::runtime_error;
using std::to_string;
using std::string;


static size_t consumeCompletions(const QueuePtr& queue)
{
    nvm_queue_t* cq = &queue->cq;
    nvm_queue_t* sq = &queue->sq;

    nvm_cpl_t* cpl = nullptr;
    size_t numCpls = 0;

    while ((cpl = nvm_cq_dequeue(cq)) != nullptr)
    {
        nvm_sq_update(sq);

        if (!NVM_ERR_OK(cpl))
        {
            throw error("Queue " + to_string(queue->no) + " got error: " + string(nvm_strerror(NVM_ERR_STATUS(cpl))));
        }

        ++numCpls;
    }

    if (numCpls > 0)
    {
        nvm_cq_update(cq);
    }

    return numCpls;
}


static void flush(const QueuePtr& queue, uint32_t ns)
{
    nvm_cmd_t local;
    memset(&local, 0, sizeof(local));
    nvm_cmd_header(&local, NVM_DEFAULT_CID(&queue->sq), NVM_IO_FLUSH, ns);
    
    nvm_cmd_t* cmd = nvm_sq_enqueue(&queue->sq);
    if (cmd == nullptr)
    {
        throw error("Queue is full, this should not happen!");
    }

    *cmd = local;

    nvm_sq_submit(&queue->sq);

    while (nvm_cq_dequeue(&queue->cq) == nullptr)
    {
        std::this_thread::yield();
    }

    nvm_sq_update(&queue->sq);
    nvm_cq_update(&queue->cq);
}



static Event sendWindow(const TransferPtr& transfer, ChunkPtr& from, const ChunkPtr& to, uint32_t ns, bool write)
{
    size_t numCmds = 0;
    size_t numBlocks = 0;
    nvm_cmd_t local;

    const QueuePtr& queue = transfer->queue;
    const DmaPtr& queueMemory = queue->getQueueMemory();
    const DmaPtr& buffer = transfer->buffer->buffer;

    memset(&local, 0, sizeof(local));

    // Fill up queue with commands
    for (numCmds = 0; numCmds < queue->depth && from != to; ++numCmds, ++from)
    {
        nvm_cmd_t* cmd = nvm_sq_enqueue(&queue->sq);
        if (cmd == nullptr)
        {
            throw error("Queue is full, this should not happen!");
        }

        const Chunk& chunk = *from;
        void* prpListPtr = NVM_DMA_OFFSET(queueMemory, 1 + numCmds);
        uint64_t prpListAddr = queueMemory->ioaddrs[1 + numCmds];

        // Build command locally
        nvm_cmd_header(&local, NVM_DEFAULT_CID(&queue->sq) * queue->no, write ? NVM_IO_WRITE : NVM_IO_READ, ns);
        nvm_cmd_rw_blks(&local, chunk.startBlock, chunk.numBlocks);
        nvm_cmd_data(&local, buffer->page_size, chunk.numPages, prpListPtr, prpListAddr, &buffer->ioaddrs[chunk.startPage]);

        // Write command to remote memory in one go (write-combining)
        *cmd = local;

        numBlocks += chunk.numBlocks;
    }

    auto before = std::chrono::high_resolution_clock::now();
    nvm_sq_submit(&queue->sq);
    std::this_thread::yield();

    // Wait for all completions
    for (size_t i = 0; i < numCmds; i += consumeCompletions(queue))
    {
        std::this_thread::yield();
    }

    auto after = std::chrono::high_resolution_clock::now();

    return Event(numCmds, numBlocks, after - before);
}



static void measureLatency(const TransferPtr& transfer, const Settings& settings, const EventListPtr& events, Barrier* barrier, bool write)
{
    for (size_t i = 0; i < settings.repeat; ++i)
    {
        const ChunkPtr end = transfer->chunks.cend();
        ChunkPtr ptr = transfer->chunks.cbegin();

        // Sync with other threads
        barrier->wait();

        while (ptr != end)
        {
            auto event = sendWindow(transfer, ptr, end, settings.nvmNamespace, write);

            events->push_back(event);
        }

        barrier->wait();

        if (write)
        {
            flush(transfer->queue, settings.nvmNamespace);
        }
    }
}



static void measureBandwidth(const TransferPtr& transfer, const Settings& settings, const EventListPtr& events, Barrier* barrier, bool write)
{
    nvm_queue_t* sq = &transfer->queue->sq;
    nvm_cmd_t local;
    memset(&local, 0, sizeof(local));

    const QueuePtr& queue = transfer->queue;
    const DmaPtr& sqMemory = queue->getQueueMemory();
    const DmaPtr& buffer = transfer->buffer->buffer;
    auto op = write ? NVM_IO_WRITE : NVM_IO_READ;
    auto ns = settings.nvmNamespace;

    for (size_t i = 0; i < settings.repeat; ++i)
    {
        size_t numCmds = 0;
        size_t numCpls = 0;
        size_t totalCmds = 0;
        size_t totalCpls = 0;
        size_t numBlocks = 0;

        // Synch with other threads
        barrier->wait();

        auto before = std::chrono::high_resolution_clock::now();

        for (const auto& chunk: transfer->chunks)
        {
            nvm_cmd_t* cmd = nullptr;

            // Queue is full, submit what we have and wait
            while (numCmds == queue->depth || (cmd = nvm_sq_enqueue(sq)) == nullptr)
            {
                nvm_sq_submit(sq);
                std::this_thread::yield();
                numCpls = consumeCompletions(queue);
                numCmds -= numCpls;
                totalCpls += numCpls;
            }

            void* prpListPtr = NVM_DMA_OFFSET(sqMemory, 1 + numCmds);
            uint64_t prpListAddr = sqMemory->ioaddrs[1 + numCmds];

            // Construct command locally
            nvm_cmd_header(&local, NVM_DEFAULT_CID(sq), op, ns);
            nvm_cmd_rw_blks(&local, chunk.startBlock, chunk.numBlocks);
            nvm_cmd_data(&local, buffer->page_size, chunk.numPages, prpListPtr, prpListAddr, &buffer->ioaddrs[chunk.startPage]);

            // Write local command to remote memory pointer in one go (write combining)
            *cmd = local;

            numBlocks += chunk.numBlocks;
            
            ++numCmds;
            ++totalCmds;
        }

        nvm_sq_submit(sq);

        // Wait until everything is completed
        while (totalCpls != totalCmds)
        {
            std::this_thread::yield();
            numCpls = consumeCompletions(queue);
            totalCpls += numCpls;
            numCmds -= numCpls;
        }

        auto after = std::chrono::high_resolution_clock::now();
        events->emplace_back(totalCmds, numBlocks, after - before);

        barrier->wait();

        if (write)
        {
            flush(queue, ns);
        }
    }
}



void benchmark(EventMap& times, const TransferMap& transfers, const Settings& settings, bool write)
{
    thread threads[transfers.size()];
    //Barrier barrier(transfers.size() + 1);
    Barrier barrier(transfers.size());

    size_t totalBlocks = 0;
    size_t totalChunks = 0;
    size_t i = 0;
    for (const auto& t: transfers)
    {
        const auto idx = t.first;
        const auto& transfer = t.second;

        auto events = std::make_shared<EventList>();

        totalBlocks += transfer->count;
        totalChunks += transfer->chunks.size();

        if (settings.latency)
        {
            threads[i++] = thread([transfer, settings, events, &barrier, write]() {
                measureLatency(transfer, settings, events, &barrier, write);
            });
        }
        else
        {
            threads[i++] = thread([transfer, settings, events, &barrier, write]() {
                measureBandwidth(transfer, settings, events, &barrier, write);
            });
        }

        times.insert(EventMap::value_type(idx, events));
    }

    fprintf(stderr, "Running %s benchmark (%s, %s)... ", 
            settings.latency ? "latency" : "bandwidth", write ? "writing" : "reading", settings.random ? "random" : "sequential");

//    // Create aggregated statistics
//    auto aggregated = std::make_shared<EventList>();
//    for (size_t i = 0; i < settings.repeat; ++i)
//    {
//        barrier.wait();
//        auto before = std::chrono::high_resolution_clock::now();
//        barrier.wait();
//        auto after = std::chrono::high_resolution_clock::now();
//
//        if (!settings.latency)
//        {
//            aggregated->emplace_back(Event(totalChunks, totalBlocks, after - before));
//        }
//    }
//
//    if (!settings.latency)
//    {
//        times.insert(EventMap::value_type(0, aggregated));
//    }

    // Wait for all threads to complete
    for (size_t i = 0; i < transfers.size(); ++i)
    {
        threads[i].join();
    }

    fprintf(stderr, "DONE\n");
}
