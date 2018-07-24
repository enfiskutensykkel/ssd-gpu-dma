#include "stats.h"
#include "benchmark.h"
#include "buffer.h"
#include "queue.h"
#include <nvm_util.h>
#include <sstream>
#include <iomanip>
#include <map>
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
using Percentiles = std::map<double, double>;



static void percentiles(Percentiles& percentiles, vector<double>& data, const vector<double>& kth)
{
    sort(data.begin(), data.end(), std::less<double>());

    for (auto k: kth)
    {
        double i = k * data.size();
        double idx = ceil(i);
        double p = data[idx];

        if (i == idx && idx < data.size())
        {
            p = (data[idx] + data[idx+1]) / 2.0;
        }

        percentiles.insert(Percentiles::value_type(k, p));
    }

    percentiles[1] = data.back();
    percentiles[0] = data.front();
}



static void printSummary(const Ctrl& ctrl, uint16_t queueNo, const EventList& events, bool write)
{
    const size_t blockSize = ctrl.blockSize;
    const size_t blocksPerPrp = NVM_PAGE_TO_BLOCK(ctrl.pageSize, blockSize, 1);

    vector<double> kth{.99, .97, .95, .90, .75, .50, .25, .10, .05, .01};
    vector<double> data;
    data.reserve(events.size());

    // Calculate bandwidth
    Percentiles bandwidth;
    for (const auto& event: events)
    {
        data.push_back(event.bandwidth(blockSize));
    }
    percentiles(bandwidth, data, kth);

    // Estimate IOPS
    data.clear();
    data.reserve(events.size());
    Percentiles iops;
    for (const auto& event: events)
    {
        data.push_back(event.adjustedIops(blocksPerPrp));
    }
    percentiles(iops, data, kth);

    // Estimate average latency per command
    data.clear();
    data.reserve(events.size());
    Percentiles cmdLatency;
    for (const auto& event: events)
    {
        data.push_back(event.averageLatencyPerCommand());
    }
    percentiles(cmdLatency, data, kth);

    // Estimate average latency per PRP/page
    data.clear();
    data.reserve(events.size());
    Percentiles prpLatency;
    for (const auto& event: events)
    {
        data.push_back(event.averageLatencyPerBlock() * blocksPerPrp);
    }
    percentiles(prpLatency, data, kth);

    data.clear();

    if (queueNo != 0)
    {
        fprintf(stderr, "Queue #%02u %s percentiles (%zu samples)\n", queueNo, write ? "write" : "read", events.size());
    }
    else
    {
        fprintf(stderr, "Aggregated %s percentiles (%zu samples)\n", write ? "write" : "read", events.size());
    }

    fprintf(stderr, "       %14s, %14s, %14s, %14s\n", "bandwidth", "adj iops", "cmd latency", "prp latency");
    fprintf(stderr, "  max: %14.3f, %14.3f, %14.3f, %14.3f\n",
            bandwidth[1], iops[1], cmdLatency[1], prpLatency[1]);

    for (auto k: kth)
    {
        fprintf(stderr, " %4.2f: %14.3f, %14.3f, %14.3f, %14.3f\n",
                k, bandwidth[k], iops[k], cmdLatency[k], prpLatency[k]);
    }

    fprintf(stderr, "  min: %14.3f, %14.3f, %14.3f, %14.3f\n",
            bandwidth[0], iops[0], cmdLatency[0], prpLatency[0]);
}



void calculatePercentiles(const Ctrl& ctrl, const EventMap& events, bool write)
{
    if (events.empty())
    {
        return;
    }

    fprintf(stderr, "Calculating percentiles...\n");
    for (const auto& ep: events)
    {
        const auto queueNo = ep.first;
        if (queueNo == 0)
        {
            continue;
        }

        printSummary(ctrl, queueNo, *ep.second, write);
    }

    if (events.size() > 2 && events.find(0) != events.end())
    {
        printSummary(ctrl, 0, *events.at(0), write);
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



static void showTransferMetadata(FILE* fp, const TransferMap& transfers, const Settings& settings, const Ctrl& ctrl)
{
    // Print queue metadata
    for (const auto& tp: transfers)
    {
        const auto queueNo = tp.first;
        const auto& queue = tp.second->queue;

        fprintf(fp, "### queue[%x]: no=%x; cmds=%zu; prps=%zu; transfer-size=%zu; remote=%s; memory=%s;\n",
                queueNo,
                queue->no,
                queue->depth,
                queue->pages,
                queue->pages * ctrl.pageSize,
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

    string ctrlString;
    if (!settings.ctrl.path.empty())
    {
        ctrlString = "'" + settings.ctrl.path + "'";
    }
    else
    {
        std::ostringstream s;
        s << std::hex << ctrl.fdid;
        ctrlString = s.str();
    }

    fprintf(fp, "### benchmark: ctrl=%s; block-size=%zu; prp-size=%zu; type=%s; write=%s; verified=%s; parallel=%s; shared=%s; random=%s; repeat=%zu; iterations=%zu\n",
            ctrlString.c_str(),
            ctrl.blockSize,
            ctrl.pageSize,
            settings.latency ? "latency" : "bandwidth",
            settings.write ? "true" : "false",
            settings.verify ? "true" : "false",
            settings.parallel ? "true" : "false",
            (!settings.parallel || settings.shared) ? "true" : "false",
            settings.random ? "true" : "false",
            settings.innerIterations,
            settings.outerIterations);
}



static void printRecords(FILE* fp, uint16_t queueNo, const EventList& events, size_t prpSize, size_t blockSize, bool write)
{
    std::ostringstream s;
    if (queueNo != 0)
    {
        s << std::right << std::setw(4) << std::setfill(' ') << std::hex << queueNo;
    }
    else
    {
        s << std::right << std::setw(4) << std::setfill(' ') << "aggr";
    }
    string q(s.str());

    const size_t blocksPerPrp = NVM_PAGE_TO_BLOCK(prpSize, blockSize, 1);

    for (const auto& event: events)
    {
		fprintf(fp, "  %4s; %5s; %8zu; %12.3f; %12zu; %12zu; %12zu; %12.3f; %12.3f; %12.3f; %12.3f; %12.3f; %12.3f; %12.3f;\n",
				q.c_str(), write ? "write" : "read", event.commands, event.time.count(), event.blocks, event.blocks / blocksPerPrp, event.transferSize(blockSize),
				event.averageLatencyPerCommand(), event.averageLatencyPerBlock(), event.averageLatencyPerBlock() * blocksPerPrp,
				event.bandwidth(blockSize), event.estimateIops(), event.adjustedIops(), event.adjustedIops(blocksPerPrp));
    }
}



void printStatistics(const Ctrl& ctrl, const EventMap& readEvents, const EventMap& writeEvents, const TransferMap& transfers, const Settings& settings)
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

    const size_t blockSize = ctrl.blockSize;
    const size_t pageSize = ctrl.pageSize;

    if (settings.transferInfo)
    {
        showTransferMetadata(fp, transfers, settings, ctrl);
    }

    fprintf(fp, "#%5s; %5s; %8s; %12s; %12s; %12s; %12s; %12s; %12s; %12s; %12s; %12s; %12s; %12s;\n",
          "queue", "rd/wr", "commands", "time", "blocks", "prps", "size", "cmd-latency", "blk-latency", "prp-latency", "bandwidth", "iops", "blks-per-sec", "prps-per-sec");
    for (const auto& tp: transfers)
    {
        const auto queueNo = tp.first;

        if (settings.write)
        {
            printRecords(fp, queueNo, *writeEvents.at(queueNo), pageSize, blockSize, true);
        }

        printRecords(fp, queueNo, *readEvents.at(queueNo), pageSize, blockSize, false);
        fflush(fp);
    }

    if (readEvents.size() > 2)
    {
        if (settings.write && writeEvents.find(0) != writeEvents.end())
        {
            printRecords(fp, 0, *writeEvents.at(0), pageSize, blockSize, true);
        }

        if (readEvents.find(0) != readEvents.end())
        {
            printRecords(fp, 0, *readEvents.at(0), pageSize, blockSize, false);
        }
    }

    fclose(fp);
}

