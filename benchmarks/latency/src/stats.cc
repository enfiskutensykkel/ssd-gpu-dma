#include "stats.h"
#include "benchmark.h"
#include "buffer.h"
#include "queue.h"
#include <set>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <string>
#include <cstddef>
#include <cstring>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cmath>

using error = std::runtime_error;
using std::string;
using std::to_string;
using std::sort;
using std::vector;


static double percentile(const vector<double>& data, double k)
{
    double i = k * data.size();
    double idx = ceil(i);

    double p = data[idx];
    if (i == idx)
    {
        p = (data[idx] + data[idx+1]) / 2.0;
    }

    return p;
}



static void printSummary(vector<double>& data)
{
    sort(data.begin(), data.end(), std::less<double>());
    fprintf(stderr, "count: %14zu\n", data.size());
    fprintf(stderr, "  max: %14.3f\n", data.back());

    for (auto k: {.99, .97, .95, .90, .75, .50, .25, .10, .05, .01})
    {
        fprintf(stderr, " %4.2f: %14.3f\n", k, percentile(data, k));
    }
    fprintf(stderr, "  min: %14.3f\n", data.front());
}



void printPercentiles(const EventMap& events, const QueueMap& queues, bool latency, bool write)
{
    if (events.empty())
    {
        return;
    }

    size_t blockSize = 0;

    fprintf(stderr, "Calculating percentiles...\n");
    for (const auto& qp: queues)
    {
        const auto queueNo = qp.first;
        const auto& eventList = events.at(queueNo);
        const auto& queue = queues.at(queueNo);
        blockSize = queue->getController().blockSize;

        vector<double> data;
        data.reserve(eventList->size());

        for (const Event& event: *eventList)
        {
            if (!latency)
            {
                data.push_back(event.bandwidth(blockSize));
            }
            else
            {
                data.push_back(event.averageUsecs());
            }
        }

        fprintf(stderr, "Queue #%02u %s percentiles (%s)\n", 
                queueNo, write ? "write" : "read", latency ? "microseconds" : "MB/s");
        printSummary(data);

    }

    if (!latency && events.find(0) != events.end())
    {
        vector<double> data;
        for (const Event& event: *events.at(0))
        {
            data.push_back(event.bandwidth(blockSize));
        }
        fprintf(stderr, "Aggregated %s percentiles (MB/s)\n", write ? "write" : "read");
        printSummary(data);
    }

    fprintf(stderr, "End percentiles\n");
}



static string memlocString(const MemoryBufferPtr& buffer)
{
    const GpuBuffer* gpuBuffer = dynamic_cast<const GpuBuffer*>(buffer.get());
    if (gpuBuffer != nullptr)
    {
        return "gpu[" + to_string(gpuBuffer->gpu->cudaDevice) + "]";
    }

    return "ram";
}



static string memlocString(const QueuePtr& queue)
{
    if (queue->location() == QueueLocation::GPU)
    {
        const GpuQueue* gpuQueue = dynamic_cast<const GpuQueue*>(queue.get());
        return "gpu[" + to_string(gpuQueue->getGpu().cudaDevice) + "]";
    }

    return "ram";
}



static void printGpuMetadata(FILE* fp, const Gpu& gpu)
{
    fprintf(fp, "### gpu[%d]: device=%d; fdid=%lx; bdf=%s; name='%s';\n",
        gpu.cudaDevice,
        gpu.cudaDevice,
        gpu.fdid,
        gpu.deviceBdf().c_str(),
        gpu.deviceName().c_str());
}



static void showTransferMetadata(FILE* fp, const TransferMap& transfers, const Settings& settings)
{
    // Print queue metadata
    for (const auto& tp: transfers)
    {
        const auto queueNo = tp.first;
        const auto& queue = tp.second->queue;

        fprintf(fp, "### queue[%x]: no=%x; cmds=%zu; prps=%zu; remote=%s; memory=%s;\n",
                queueNo,
                queue->no,
                queue->depth,
                queue->pages,
                queue->location() != QueueLocation::LOCAL ? "true" : "false",
                memlocString(queue).c_str());
    }

    // Print GPU metadata
    std::set<int> gpus;
    for (const auto& tp: transfers)
    {
        const GpuQueue* queue = dynamic_cast<const GpuQueue*>(tp.second->queue.get());
        if (queue != nullptr && gpus.find(queue->getGpu().cudaDevice) == gpus.end())
        {
            const auto& gpu = queue->getGpu();
            gpus.insert(gpu.cudaDevice);
            printGpuMetadata(fp, gpu);
        }

        const GpuBuffer* buffer = dynamic_cast<const GpuBuffer*>(tp.second->buffer.get());
        if (buffer != nullptr && gpus.find(buffer->gpu->cudaDevice) == gpus.end())
        {
            const auto& gpu = *buffer->gpu;
            gpus.insert(gpu.cudaDevice);
            printGpuMetadata(fp, gpu);
        }
    }

    // Print transfer metadata
    for (const auto& tp: transfers)
    {
        const auto queueNo = tp.first;
        const auto& transfer = tp.second;

        fprintf(fp, "### transfer[%x]: queue=%x; pattern=%s; buffer=%s; buffer-prps=%zu; prp-offset=%zu; block-count=%zu; block-offset=%zu; cmds=%zu;\n",
                queueNo,
                transfer->queue->no,
                settings.random ? "random" : "sequential",
                memlocString(transfer->buffer).c_str(), 
                transfer->buffer->buffer->n_ioaddrs,
                transfer->pageOffset,
                transfer->count, 
                transfer->offset,
                transfer->chunks.size());
    }

    fprintf(fp, "### benchmark: type=%s; write=%s; verified=%s; parallel=%s; shared=%s; random=%s; repetitions=%zu;\n",
            settings.latency ? "latency" : "bandwidth",
            settings.write ? "true" : "false",
            settings.verify ? "true" : "false",
            settings.parallel ? "true" : "false",
            (!settings.parallel || settings.shared) ? "true" : "false",
            settings.random ? "true" : "false",
            settings.repeat);
}



static void printRecords(FILE* fp, uint16_t queueNo, const EventList& events, size_t blockSize, bool write)
{
    for (const auto& event: events)
    {
        fprintf(fp, "  %4x; %5s; %8zu; %12.3f; %12zu; %12zu; %12.3f; %12.3f; %12.3f;\n",
                queueNo, write ? "write" : "read", event.commands, event.time.count(), event.blocks,
                event.transferSize(blockSize), event.averageUsecs(), event.bandwidth(blockSize),
                event.estimateIops());
    }
}



void printStatistics(const EventMap& readEvents, const EventMap& writeEvents, const TransferMap& transfers, const Settings& settings)
{
    FILE* fp = stdout;
    if (!settings.statsFilename.empty())
    {
        fp = fopen(settings.statsFilename.c_str(), "w");
        if (fp == nullptr)
        {
            throw error("Failed to open statistics file: " + string(strerror(errno)));
        }
    }

    if (settings.transferInfo)
    {
        showTransferMetadata(fp, transfers, settings);
    }

    fprintf(fp, "#%5s; %5s; %8s; %12s; %12s; %12s; %12s; %12s; %12s;\n",
          "queue", "rwdir", "cmds", "usecs", "blocks", "size", "lat", "bw", "iops");
    size_t estBlockSize = 0;
    for (const auto& tp: transfers)
    {
        const auto queueNo = tp.first;
        const auto blockSize = tp.second->queue->getController().blockSize;
        estBlockSize = blockSize;

        if (settings.write)
        {
            printRecords(fp, queueNo, *writeEvents.at(queueNo), blockSize, true);
        }

        printRecords(fp, queueNo, *readEvents.at(queueNo), blockSize, false);
    }

    if (!settings.latency)
    {
        if (settings.write)
        {
            printRecords(fp, 0, *writeEvents.at(0), estBlockSize, true);
        }

        printRecords(fp, 0, *readEvents.at(0), estBlockSize, false);
    }

    fclose(fp);
}

