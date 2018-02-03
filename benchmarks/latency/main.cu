#include "settings.h"
#include "buffer.h"
#include "ctrl.h"
#include "queue.h"
#include "barrier.h"
#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_error.h>
#include <nvm_util.h>
#include <nvm_queue.h>
#include <nvm_cmd.h>
#include <stdexcept>
#include <vector>
#include <memory>
#include <algorithm>
#include <functional>
#include <thread>
#include <chrono>
#include <string>
#include <limits>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <sisci_api.h>

using std::string;
using std::runtime_error;
using std::make_shared;
using std::thread;


typedef std::chrono::duration<double, std::micro> mtime;

struct Time
{
    uint16_t    commands;
    size_t      blocks;
    mtime       time;
    
    Time(uint16_t commands, size_t blocks, mtime time)
        : commands(commands), blocks(blocks), time(time) {}
};


typedef std::vector<Time> Times;


static size_t createQueues(const Controller& ctrl, Settings& settings, QueueList& queues)
{
    const size_t pageSize = ctrl.info.page_size;
    const size_t blockSize = ctrl.ns.lba_data_size;

    const size_t transferPages = NVM_PAGE_ALIGN(settings.numBlocks * blockSize, pageSize) / pageSize;
    const size_t pagesPerQueue = transferPages / ctrl.numQueues;

    srand(settings.startBlock);

    const size_t maxDataBlock = NVM_PAGE_TO_BLOCK(pageSize, blockSize, ctrl.info.max_data_pages);
    const size_t maxBlock = ctrl.ns.size / blockSize;

    size_t dataPages = 0;

    bool write = settings.write;

    for (uint16_t i = 0; i < ctrl.numQueues; ++i)
    {
        auto queue = make_shared<Queue>(ctrl, settings.adapter, settings.segmentId++, i+1, settings.queueDepth, settings.remote);
        size_t pageOff = pagesPerQueue * i;

        fprintf(stderr, "Queue #%02u %s qd=%zu ", queue->no, settings.remote ? "remote" : "local", queue->depth);
        switch (settings.pattern)
        {
            case AccessPattern::SEQUENTIAL:
                dataPages += prepareRange(queue->transfers, ctrl, dataPages, settings.startBlock, settings.numBlocks, write);
                fprintf(stderr, "blocks=%zu offset=%zu pattern=sequential", settings.numBlocks, settings.startBlock);
                break;

            case AccessPattern::LINEAR:
                if (i == ctrl.numQueues - 1)
                {
                    size_t startBlock = settings.startBlock + NVM_PAGE_TO_BLOCK(pageSize, blockSize, pageOff);
                    size_t numBlocks = settings.numBlocks - startBlock;

                    dataPages += prepareRange(queue->transfers, ctrl, pageOff, startBlock, numBlocks, write);
                    fprintf(stderr, "blocks=%zu offset=%zu pattern=linear", numBlocks, startBlock);
                }
                else
                {
                    size_t startBlock = settings.startBlock + NVM_PAGE_TO_BLOCK(pageSize, blockSize, pageOff);
                    size_t numBlocks = NVM_PAGE_TO_BLOCK(pageSize, blockSize, pagesPerQueue);
                    
                    dataPages += prepareRange(queue->transfers, ctrl, pageOff, startBlock, numBlocks, write);
                    fprintf(stderr, "blocks=%zu offset=%zu pattern=linear", numBlocks, startBlock);
                }
                break;

            //case AccessPattern::RANDOM: // FIXME: This is random-sequential
            //    dataPages += prepareRange(queue->transfers, ctrl, dataPages, rand() % maxBlock, settings.numBlocks, false);
            //    break;

            case AccessPattern::RANDOM:
                for (size_t i = 0; i < settings.numBlocks / maxDataBlock; ++i)
                {
                    dataPages += prepareRange(queue->transfers, ctrl, dataPages, rand() % maxBlock, maxDataBlock, write);
                }
                fprintf(stderr, "blocks=%zu pattern=random", (settings.numBlocks / maxDataBlock) * maxDataBlock);
                break;
        }
        fprintf(stderr, " (%zu commands)\n", queue->transfers.size());

        queues.push_back(queue);
    }

    return dataPages;
}


static void benchmark(const QueueList& queues, const BufferPtr& buffer, const Settings& settings, size_t blockSize);



//static void dumpMemory(const BufferPtr& buffer, bool ascii)
//{
//    uint8_t* ptr = (uint8_t*) buffer->vaddr;
//    size_t byte = 0;
//    size_t size = buffer->page_size * buffer->n_ioaddrs;
//    while (byte < size)
//    {
//        fprintf(stderr, "%8lx: ", byte);
//        for (size_t n = byte + (ascii ? 0x80 : 0x20); byte < n; ++byte)
//        {
//            uint8_t value = ptr[byte];
//            if (ascii)
//            {
//                if ( !(0x20 <= value && value <= 0x7e) )
//                {
//                    value = ' ';
//                }
//                fprintf(stdout, "%c", value);
//            }
//            else
//            {
//                fprintf(stdout, " %02x", value);
//            }
//        }
//        fprintf(stdout, "\n");
//    }
//}


static void verify(const Controller& ctrl, const QueueList& queues, const BufferPtr& buffer, const Settings& settings)
{
    size_t fileSize = settings.numBlocks * ctrl.ns.lba_data_size;

    if (settings.write)
    {
        throw runtime_error("Unable to verify written data");
    }

    void* ptr = malloc(fileSize);
    if (ptr == nullptr)
    {
        throw runtime_error(string("Failed to allocate local buffer: ") + strerror(errno));
    }

    FILE* fp = fopen(settings.filename, "r");
    if (fp == nullptr)
    {
        free(ptr);
        throw runtime_error(string("Failed to open file: ") + strerror(errno));
    }

    size_t actualSize = fread(ptr, 1, fileSize, fp);
    fclose(fp);

    if (actualSize != fileSize)
    {
        fprintf(stderr, "WARNING: Verification file differs in size!\n");
    }

    void* bufferPtr = buffer->vaddr;
    if (settings.cudaDevice != -1)
    {
        cudaHostAlloc(&bufferPtr, fileSize, cudaHostAllocDefault);
        cudaMemcpy(bufferPtr, buffer->vaddr, actualSize, cudaMemcpyDeviceToHost);
    }

    switch (settings.pattern)
    {
        case AccessPattern::SEQUENTIAL:
            for (const auto& queue: queues)
            {
                const auto& start = *queue->transfers.begin();

                //if (memcmp(ptr, NVM_DMA_OFFSET(buffer, start.startPage), actualSize) != 0)
                if (memcmp(ptr, NVM_PTR_OFFSET(bufferPtr, buffer->page_size, start.startPage), actualSize) != 0)
                {
                    free(ptr);
                    if (settings.cudaDevice != -1)
                    {
                        cudaFree(bufferPtr);
                    }
                    throw runtime_error("File differs!");
                }
            }
            break;

        case AccessPattern::LINEAR:
            if (memcmp(ptr, bufferPtr, actualSize) != 0)
            {
                free(ptr);
                if (settings.cudaDevice != -1)
                {
                    cudaFree(bufferPtr);
                }
                throw runtime_error("File differs!");
            }
            break;

        case AccessPattern::RANDOM:
            free(ptr);
            if (settings.cudaDevice != -1)
            {
                cudaFree(bufferPtr);
            }
            throw runtime_error("Unable to verify random blocks!");
    }

    free(ptr);
    if (settings.cudaDevice != -1)
    {
        cudaFree(bufferPtr);
    }
}



int main(int argc, char** argv)
{
    Settings settings;

    // Parse command line arguments
    try
    {
        settings.parseArguments(argc, argv);
    }
    catch (const string& s)
    {
        fprintf(stderr, "%s\n", s.c_str());
        return 1;
    }

    sci_error_t err;
    SCIInitialize(0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Something went wrong: %s\n", SCIGetErrorString(err));
        return 1;
    }

    try
    {
        fprintf(stderr, "Resetting controller...\n");
#ifdef __DIS_CLUSTER__
        Controller ctrl(settings.controllerId, settings.adapter, settings.segmentId++, settings.nvmNamespace, settings.numQueues);
#else
        Controller ctrl(settings.controllerPath, settings.nvmNamespace, settings.numQueues);
#endif

        settings.numQueues = ctrl.numQueues;

        QueueList queues;
        size_t numPages = createQueues(ctrl, settings, queues);

        fprintf(stderr, "Creating buffer (%zu pages)...\n", numPages);
        BufferPtr buffer = createBuffer(ctrl.ctrl, settings.adapter, settings.segmentId++, numPages * ctrl.ctrl->page_size, settings.cudaDevice);

        benchmark(queues, buffer, settings, ctrl.ns.lba_data_size);

        if (settings.filename != nullptr && settings.pattern != AccessPattern::RANDOM)
        {
            fprintf(stderr, "Verifying transfer...\n");
            verify(ctrl, queues, buffer, settings);
        }

        //dumpMemory(buffer, false);
    }
    catch (const runtime_error& e)
    {
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }

    fprintf(stderr, "OK!\n");
    SCITerminate();
    return 0;
}




static Time sendWindow(QueuePtr& queue, TransferPtr& from, const TransferPtr& to, const BufferPtr& buffer, uint32_t ns, Barrier* barrier)
{
    size_t numCommands = 0;
    size_t numBlocks = 0;

    // Fill up to queue depth with commands
    for (numCommands = 0; numCommands < queue->depth && from != to; ++numCommands, ++from)
    {
        nvm_cmd_t* cmd = nvm_sq_enqueue(&queue->sq);
        if (cmd == nullptr)
        {
            throw runtime_error(string("Queue is full, should not happen!"));
        }

        const Transfer& t = *from;
        void* prpListPtr = NVM_DMA_OFFSET(queue->sq_mem, 1 + numCommands);
        uint64_t prpListAddr = queue->sq_mem->ioaddrs[1 + numCommands];
        
        nvm_cmd_header(cmd, t.write ? NVM_IO_WRITE : NVM_IO_READ, ns);
        nvm_cmd_rw_blks(cmd, t.startBlock, t.numBlocks);
        nvm_cmd_data(cmd, buffer->page_size, t.numPages, prpListPtr, prpListAddr, &buffer->ioaddrs[t.startPage]);

        numBlocks += t.numBlocks;
    }

    // Sync with other threads
    barrier->wait();

    // Get current time before submitting
    auto before = std::chrono::high_resolution_clock::now();
    nvm_sq_submit(&queue->sq);
    std::this_thread::yield();

    // Wait for all completions
    for (size_t i = 0; i < numCommands; ++i)
    {
        nvm_cpl_t* cpl;
        while ((cpl = nvm_cq_dequeue(&queue->cq)) == nullptr)
        {
            std::this_thread::yield();
        }

        nvm_sq_update(&queue->sq);

        if (!NVM_ERR_OK(cpl))
        {
            fprintf(stderr, "%u: %s\n", queue->no, nvm_strerror(NVM_ERR_STATUS(cpl)));
        }

        nvm_cq_update(&queue->cq);
    }

    // Get current time after all commands completed
    auto after = std::chrono::high_resolution_clock::now();

    return Time(numCommands, numBlocks, after - before);
}



static void flush(QueuePtr& queue, uint32_t ns)
{
    nvm_cmd_t* cmd = nvm_sq_enqueue(&queue->sq);
    if (cmd == nullptr)
    {
        throw runtime_error(string("Queue is full, should not happen!"));
    }

    nvm_cmd_header(cmd, NVM_IO_FLUSH, ns);
    nvm_cmd_data_ptr(cmd, 0, 0);

    nvm_sq_submit(&queue->sq);

    while (nvm_cq_dequeue(&queue->cq) == nullptr)
    {
        std::this_thread::yield();
    }
    nvm_sq_update(&queue->sq);
    nvm_cq_update(&queue->cq);
}



static void measure(QueuePtr queue, const BufferPtr buffer, Times* times, const Settings& settings, Barrier* barrier)
{
    for (size_t i = 0; i < settings.repetitions; ++i)
    {
        const TransferPtr transferEnd = queue->transfers.cend();
        TransferPtr transferPtr = queue->transfers.cbegin();
        
        while (transferPtr != transferEnd)
        {
            auto time = sendWindow(queue, transferPtr, transferEnd, buffer, settings.nvmNamespace, barrier);

            times->push_back(time);
        }

        flush(queue, settings.nvmNamespace);
    }
}



static double percentile(const std::vector<double>& values, double p)
{
    double index = ceil(p * values.size());
    return values[index];
}
    


static void printStatistics(const QueuePtr& queue, const Times& times, size_t blockSize, bool print)
{
    double minLat = std::numeric_limits<double>::max();
    double maxLat = std::numeric_limits<double>::min();
    double avgLat = 0;

    size_t blocks = 0;

    std::vector<double> latencies;
    latencies.reserve(times.size());

    for (const auto& t: times)
    {
        const auto current = t.time.count();

        if (current < minLat)
        {
            minLat = current;
        }
        else if (current > maxLat)
        {
            maxLat = current;
        }

        avgLat += current;
        latencies.push_back(current);

        blocks += t.blocks;

        if (print)
        {
            double bw = (t.blocks * blockSize) / current; 
            fprintf(stdout, "#%04x %8u %12zu %12.3f %12.3f\n",
                    queue->no, t.commands, t.blocks, current, bw);
        }
    }

    avgLat /= times.size();


    fprintf(stderr, "Queue #%02u total-blocks=%zu count=%zu ",
            queue->no, blocks, times.size());
    fprintf(stderr, "min=%.3f avg=%.3f max=%.3f\n", minLat, avgLat, maxLat);

    // Calculate percentiles
    std::sort(latencies.begin(), latencies.end(), std::greater<double>());
    std::reverse(latencies.begin(), latencies.end());

    for (auto p: {.99, .97, .95, .90, .75, .50, .25, .05, .01})
    {
        fprintf(stderr, "\t%4.2f: %14.3f\n", p, percentile(latencies, p));
    }
}



static void benchmark(const QueueList& queues, const BufferPtr& buffer, const Settings& settings, size_t blockSize)
{
    Times times[queues.size()];
    thread threads[queues.size()];

    if (settings.cudaDevice == -1)
    {
        memset(buffer->vaddr, 0x00, buffer->page_size * buffer->n_ioaddrs);
    }
    else
    {
        cudaMemset(buffer->vaddr, 0x00, buffer->page_size * buffer->n_ioaddrs);
    }

    Barrier barrier(queues.size());

    for (size_t i = 0; i < queues.size(); ++i)
    {
        Times* t = &times[i];
        QueuePtr q = queues[i];

        //threads[i] = thread(measure, &queues[i], &buffer, &times[i], &settings, &barrier);
        threads[i] = thread([&q, &buffer, t, &settings, &barrier] {
            measure(q, buffer, t, settings, &barrier);
        });
    }

    fprintf(stderr, "Running benchmark...\n");

    for (size_t i = 0; i < queues.size(); ++i)
    {
        threads[i].join();
        printStatistics(queues[i], times[i], blockSize, settings.stats);
    }
}

