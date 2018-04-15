#include "settings.h"
#include "benchmark.h"
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
#ifdef __DIS_CLUSTER__
#include <sisci_api.h>
#endif

using std::string;
using std::runtime_error;
using std::make_shared;
using std::thread;



static string patternString(const Settings& settings)
{
    switch (settings.pattern)
    {
        case AccessPattern::LINEAR:
            return "linear";

        case AccessPattern::SEQUENTIAL:
            return "sequential";

        case AccessPattern::RANDOM_LINEAR:
            return "random-offset";

        case AccessPattern::RANDOM_CHUNK:
            return "random-chunks";

        case AccessPattern::RANDOM_PAGE:
            return "random-pages";
    }

    throw runtime_error("Unknown access pattern");
}



static size_t createQueues(const Controller& ctrl, Settings& settings, QueueList& queues)
{
    const size_t pageSize = ctrl.info.page_size;
    const size_t blockSize = ctrl.ns.lba_data_size;

    size_t pages = 0;
    size_t totalPages = NVM_PAGE_ALIGN(settings.numBlocks * blockSize, pageSize) / pageSize;

    srand(settings.startBlock);

    for (uint16_t no = 1; no <= ctrl.numQueues; ++no)
    {
#ifdef __DIS_CLUSTER__
        auto queue = make_shared<Queue>(ctrl, settings.adapter, settings.segmentId++, no, settings.queueDepth, settings.remote);
#else
        auto queue = make_shared<Queue>(ctrl, no, settings.queueDepth);
#endif

        switch (settings.pattern)
        {
            case AccessPattern::LINEAR:
                pages += prepareRange(queue->transfers, ctrl, settings.write, pages, settings.startBlock, settings.numBlocks);
                break;

            case AccessPattern::SEQUENTIAL:
                if (no == ctrl.numQueues)
                {
                    auto start = settings.startBlock + NVM_PAGE_TO_BLOCK(pageSize, blockSize, pages);
                    auto blocks = settings.numBlocks - NVM_PAGE_TO_BLOCK(pageSize, blockSize, pages);
                    pages += prepareRange(queue->transfers, ctrl, settings.write, pages, start, blocks);
                }
                else
                {
                    auto start = settings.startBlock + NVM_PAGE_TO_BLOCK(pageSize, blockSize, pages);
                    auto blocks = NVM_PAGE_TO_BLOCK(pageSize, blockSize, totalPages / ctrl.numQueues);
                    pages += prepareRange(queue->transfers, ctrl, settings.write, pages, start, blocks);
                }
                break;

            case AccessPattern::RANDOM_LINEAR:
                pages += fillRandom(queue->transfers, ctrl, settings.write, settings.numBlocks, false, false);
                break;

            case AccessPattern::RANDOM_CHUNK:
                pages += fillRandom(queue->transfers, ctrl, settings.write, settings.numBlocks, true, false);
                break;

            case AccessPattern::RANDOM_PAGE:
                pages += fillRandom(queue->transfers, ctrl, settings.write, settings.numBlocks, true, true);
                break;
        }

        if (queue->transfers.empty())
        {
            throw runtime_error("Transfer size is too small for more than one queue");
        }

        fprintf(stderr, "Queue #%02u %s %zu commands\n",
                no, settings.remote ? "remote" : "local", queue->transfers.size());

        queues.push_back(queue);
    }
    
    return pages;
}



static void copyMemory(const MemPtr& outputBuffer, const DmaPtr& buffer, size_t offset, size_t size, int cudaDevice)
{
    void* sourcePtr = (void*) (((unsigned char*) buffer->vaddr) + offset);

    if (cudaDevice != -1)
    {
        cudaError_t err = cudaMemcpy(outputBuffer.get(), sourcePtr, size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            throw runtime_error(string("Failed to copy memory from device: ") + cudaGetErrorString(err));
        }
    }
    else
    {
        memcpy(outputBuffer.get(), sourcePtr, size);
    }
}



static void writeToFile(const MemPtr& outputBuffer, size_t size, const string& filename)
{
    FILE* fp = fopen(filename.c_str(), "wb");
    if (fp == nullptr)
    {
        throw runtime_error(string("Failed to open file: ") + strerror(errno));
    }

    fwrite(outputBuffer.get(), 1, size, fp);
    fflush(fp);
    fclose(fp);
}



static void outputFile(const Controller& ctrl, const QueueList& queues, const DmaPtr& buffer, const Settings& settings)
{
    size_t fileSize = settings.numBlocks * ctrl.ns.lba_data_size;

    if (settings.write)
    {
        throw runtime_error("Unable to verify written data");
    }

    auto outputBuffer = createHostMemory(fileSize);

    if (settings.pattern == AccessPattern::SEQUENTIAL)
    {
        copyMemory(outputBuffer, buffer, 0, fileSize, settings.cudaDevice);
        writeToFile(outputBuffer, fileSize, string(settings.filename) + "-sequential");
    }
    else if (settings.pattern == AccessPattern::LINEAR)
    {
        for (const auto& queue: queues)
        {
            size_t offset = NVM_PAGE_ALIGN(fileSize, ctrl.info.page_size) * (queue->no - 1);
            copyMemory(outputBuffer, buffer, offset, fileSize, settings.cudaDevice);
            writeToFile(outputBuffer, fileSize, string(settings.filename) + "-linear-q" + std::to_string(queue->no));
        }
    }
}



static Time sendWindow(QueuePtr& queue, TransferPtr& from, const TransferPtr& to, const DmaPtr& buffer, uint32_t ns, Barrier* barrier)
{
    size_t numCommands = 0;
    size_t numBlocks = 0;
    nvm_cmd_t local;
    memset(&local, 0, sizeof(local));

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
        
        nvm_cmd_header(&local, NVM_DEFAULT_CID(&queue->sq), t.write ? NVM_IO_WRITE : NVM_IO_READ, ns);
        nvm_cmd_rw_blks(&local, t.startBlock, t.numBlocks);
        nvm_cmd_data(&local, buffer->page_size, t.numPages, prpListPtr, prpListAddr, &buffer->ioaddrs[t.startPage]);

        // Write command to remote memory in one go (due to write combining)
        *cmd = local;

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

    nvm_cmd_header(cmd, NVM_DEFAULT_CID(&queue->sq), NVM_IO_FLUSH, ns);
    nvm_cmd_data_ptr(cmd, 0, 0);

    nvm_sq_submit(&queue->sq);

    while (nvm_cq_dequeue(&queue->cq) == nullptr)
    {
        std::this_thread::yield();
    }
    nvm_sq_update(&queue->sq);
    nvm_cq_update(&queue->cq);
}



static void measureLatency(QueuePtr queue, const DmaPtr buffer, Times* times, const Settings& settings, Barrier* barrier)
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



static size_t consumeCompletions(QueuePtr& queue)
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
            fprintf(stderr, "%u: %s\n", queue->no, nvm_strerror(NVM_ERR_STATUS(cpl)));
        }

        ++numCpls;
    }

    nvm_cq_update(cq);

    return numCpls;
}



static void measureBandwidth(QueuePtr queue, const DmaPtr buffer, Times* times, const Settings& settings, Barrier* barrier)
{
    nvm_queue_t* sq = &queue->sq;
    nvm_cmd_t local;
    memset(&local, 0, sizeof(local));

    barrier->wait();

    for (size_t i = 0; i < settings.repetitions; ++i)
    {
        size_t numCmds = 0;
        size_t numCpls = 0;
        size_t totalCmds = 0;
        size_t totalCpls = 0;
        size_t numBlocks = 0;

        auto before = std::chrono::high_resolution_clock::now();

        for (const auto& transfer: queue->transfers)
        {
            nvm_cmd_t* cmd = nullptr;

            while (numCmds == queue->depth || (cmd = nvm_sq_enqueue(sq)) == nullptr)
            {
                nvm_sq_submit(sq);
                std::this_thread::yield();
                numCpls = consumeCompletions(queue);
                numCmds -= numCpls;
                totalCpls += numCpls;
            }
            
            void* prpListPtr = NVM_DMA_OFFSET(queue->sq_mem, 1 + numCmds);
            uint64_t prpListAddr = queue->sq_mem->ioaddrs[1 + numCmds];

            nvm_cmd_header(&local, NVM_DEFAULT_CID(sq), transfer.write ? NVM_IO_WRITE : NVM_IO_READ, settings.nvmNamespace);
            nvm_cmd_rw_blks(&local, transfer.startBlock, transfer.numBlocks);
            nvm_cmd_data(&local, buffer->page_size, transfer.numPages, prpListPtr, prpListAddr, &buffer->ioaddrs[transfer.startPage]);
           
            // Command may be a pointer to remote memory,
            // so to avoid issues with write-combining we copy
            // everything in one go
            *cmd = local;

            numBlocks += transfer.numBlocks;

            ++numCmds;
            ++totalCmds;
        }

        nvm_sq_submit(sq);

        while (totalCpls != totalCmds)
        {
            std::this_thread::yield();
            numCpls = consumeCompletions(queue);
            totalCpls += numCpls;
            numCmds -= numCpls;
        }

        auto after = std::chrono::high_resolution_clock::now();
        times->push_back(Time(totalCmds, numBlocks, after - before));

        flush(queue, settings.nvmNamespace);
    }
}



static double percentile(const std::vector<double>& values, double p)
{
    double index = ceil(p * values.size());
    return values[index];
}


static void printPercentiles(std::vector<double>& data)
{
    std::sort(data.begin(), data.end(), std::greater<double>());
    std::reverse(data.begin(), data.end());

    fprintf(stderr, "count: %14zu\n", data.size());
    fprintf(stderr, "  max: %14.3f\n", data.back());
    for (auto p: {.99, .97, .95, .90, .75, .50, .25})
    {
        fprintf(stderr, " %4.2f: %14.3f\n", p, percentile(data, p));
    }
    fprintf(stderr, "  min: %14.3f\n", data.front());
}

static void printStats(const QueuePtr& queue, const Times& times, size_t blockSize, const Settings& settings)
{
    if (queue->no == 1)
    {
        fprintf(stdout, "#%5s; %6s; %4s; %13s; %8s; %8s; %12s; %12s; %12s;\n",
                "queue", "sqhost", "type", "pattern", "depth", "cmds", "blocks", "lat", "bw");
        fflush(stdout);
    }

    auto pattern = patternString(settings);
    auto patternPtr = pattern.c_str();

#ifdef __DIS_CLUSTER__
    bool local = !settings.remote;
#else
    bool local = true;
#endif

    bool gpu = settings.cudaDevice != -1;

    for (const auto& t: times)
    {
        auto time = t.time.count();
        auto blocks = t.blocks;
        
        double bw = (blocks * blockSize) / time;

        fprintf(stdout, " %5x; %6s; %4s; %13s; %8zu; %8u; %12zu; %12.3f; %12.3f;\n",
                queue->no, local ? "local" : "remote", gpu ? "gpu" : "ram", patternPtr,
                queue->depth, t.commands, t.blocks, time, bw);
        fflush(stdout);
    }
}

static void bandwidthStats(const Times& times, size_t blockSize, std::vector<double>& results)
{
    for (const auto& t: times)
    {
        const auto currentTime = t.time.count();
        const auto blocks = t.blocks;
        const double bw = (blocks * blockSize) / currentTime;
        results.push_back(bw);
    }
}


static void latencyStats(const Times& times, std::vector<double>& results)
{
    for (const auto& t: times)
    {
        results.push_back(t.time.count());
    }
}


static void benchmark(const QueueList& queues, const DmaPtr& buffer, const Settings& settings, size_t blockSize)
{
    cudaError_t err;
    Times times[queues.size()];
    thread threads[queues.size()];

    if (settings.cudaDevice != -1)
    {
        err = cudaSetDevice(settings.cudaDevice);
        if (err != cudaSuccess)
        {
            throw runtime_error(string("Failed to set CUDA device: ") + cudaGetErrorString(err));
        }

        auto size = settings.numBlocks * blockSize;
        err = cudaMemset(buffer.get(), 0, size);
    }

    Barrier barrier(queues.size());

    if (settings.latency)
    {
        for (size_t i = 0; i < queues.size(); ++i)
        {
            Times* t = &times[i];
            QueuePtr q = queues[i];

            //auto func = std::bind(measureLatency, q, buffer, t, settings, &barrier);
            //threads[i] = thread(func);
            threads[i] = thread([q, buffer, t, settings, &barrier]() {
                measureLatency(q, buffer, t, settings, &barrier);
            });
        }
    }
    else
    {
        for (size_t i = 0; i < queues.size(); ++i)
        {
            Times* t = &times[i];
            QueuePtr q = queues[i];

            threads[i] = thread([q, buffer, t, settings, &barrier]() {
                measureBandwidth(q, buffer, t, settings, &barrier);
            });
        }
    }

    fprintf(stderr, "Running benchmark... (mode=%s, pattern=%s)\n", 
            settings.latency ? "latency" : "bandwidth", patternString(settings).c_str());

    std::vector<double> all;
    for (size_t i = 0; i < queues.size(); ++i)
    {
        threads[i].join();
        if (settings.latency)
        {
            latencyStats(times[i], all);
        }
        else
        {
            bandwidthStats(times[i], blockSize, all);
        }

        if (settings.stats)
        {
            printStats(queues[i], times[i], blockSize, settings);
        }
    }

    printPercentiles(all);

    if (settings.cudaDevice != -1)
    {
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Synchronizing CUDA device failed: %s\n", cudaGetErrorString(err));
        }
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

#ifdef __DIS_CLUSTER__
    sci_error_t err;
    SCIInitialize(0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Something went wrong: %s\n", SCIGetErrorString(err));
        return 1;
    }

    sci_desc_t sd;
    SCIOpen(&sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Something went wrong: %s\n", SCIGetErrorString(err));
        return 1;
    }

    sci_device_t cudaDev;
    if (settings.cudaDeviceId)
    {
        SCIBorrowDevice(sd, &cudaDev, settings.cudaDeviceId, SCI_FLAG_EXCLUSIVE, &err);
        if (err != SCI_ERR_OK)
        {
            fprintf(stderr, "Something went wrong: %s\n", SCIGetErrorString(err));
            return 1;
        }
    }
#endif

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

        fprintf(stderr, "Allocating %zu pages (%s)...\n", 
                numPages, settings.cudaDevice >= 0 ? "GPU" : "host");

        DmaPtr buffer;
        if (settings.cudaDevice != -1)
        {
            fprintf(stderr, "Using CUDA device: %d %s (%s)\n", settings.cudaDevice, settings.cudaDeviceName.c_str(), settings.getDeviceBDF().c_str());
            buffer = createDeviceDma(ctrl.ctrl, numPages * ctrl.ctrl->page_size, settings.cudaDevice, settings.adapter, settings.segmentId++);
        }
        else
        {
            buffer = createHostDma(ctrl.ctrl, numPages * ctrl.ctrl->page_size, settings.adapter, settings.segmentId++);
        }

        benchmark(queues, buffer, settings, ctrl.ns.lba_data_size);

        if (settings.filename != nullptr && !settings.write && !isRandom(settings.pattern))
        {
            fprintf(stderr, "Writing to file...\n");
            outputFile(ctrl, queues, buffer, settings);
        }
    }
    catch (const runtime_error& e)
    {
#ifdef __DIS_CLUSTER__
        if (settings.cudaDeviceId)
        {
            SCIReturnDevice(cudaDev, 0, &err);
        }
        SCIClose(sd, 0, &err);
#endif
        fprintf(stderr, "Unexpected error: %s\n", e.what());
        return 1;
    }

    fprintf(stderr, "OK!\n");

#ifdef __DIS_CLUSTER__
    if (settings.cudaDeviceId)
    {
        SCIReturnDevice(cudaDev, 0, &err);
    }
    SCIClose(sd, 0, &err);
    SCITerminate();
#endif
    return 0;
}

