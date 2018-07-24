#ifndef __LATENCY_BENCHMARK_SETTINGS_H__
#define __LATENCY_BENCHMARK_SETTINGS_H__

#include "gpu.h"
#include "ctrl.h"
#include "queue.h"
#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>


struct CtrlParam
{
    std::string     path;
    uint64_t        fdid;
    uint32_t        adapter;
};


struct GpuParam
{
    int             device;
    uint64_t        fdid;
    uint32_t        adapter;
};


struct QueueParam
{
    uint16_t        no;
    uint16_t        depth;
    size_t          pages;
    QueueLocation   location;
    uint64_t        fdid;
    uint32_t        adapter;
    QueueParam();
    QueueParam(const std::string& arg);
};


enum AccessUnit : int
{
    BLOCKS, PAGES, BYTES, COMMANDS
};


struct Settings
{
    CtrlParam               ctrl;
    std::vector<QueueParam> queues;
    GpuParam                gpu;
    bool                    latency;
    uint32_t                nvmNamespace;
    size_t                  outerIterations;
    size_t                  innerIterations;
    size_t                  count;
    size_t                  offset;
    AccessUnit              unit;
    bool                    sequential;
    bool                    parallel;
    bool                    shared;
    bool                    random;
    std::string             outputFilename;
    bool                    stats;
    std::string             statsFilename;
    std::string             inputFilename;
    bool                    manager;
    uint32_t                segmentOffset;
    bool                    write;
    bool                    verify;
    bool                    transferInfo;

    Settings();

    uint32_t id()
    {
        return segmentOffset++;
    }

    void parseArguments(int argc, char** argv);
};


#endif /* __LATENCY_BENCHMARK_SETTINGS_H__ */
