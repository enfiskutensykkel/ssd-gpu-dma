#ifndef __SETTINGS_H__
#define __SETTINGS_H__

#include <cstddef>
#include <cstdint>
#include <string>


enum AccessPattern : int
{
    LINEAR,              // All threads read the same sequential chunk
    SEQUENTIAL,          // All threads read the same (larger) sequential chunk with different offsets
    RANDOM_LINEAR,       // All threads read linear range offset by random block count
    RANDOM_CHUNK,        // All threads read linear chunk offset by random block count
    RANDOM_PAGE,         // All threads read one page, offset by random block count
};


enum QueueLocation : int
{
    REMOTE,
    LOCAL,
    GPU
};


constexpr bool isRandom(AccessPattern p)
{
    return p != AccessPattern::LINEAR && p != AccessPattern::SEQUENTIAL;
}


struct Settings
{
    bool            latency;
    int             cudaDevice;
    uint64_t        cudaDeviceId;
    const char*     controllerPath;
    uint64_t        controllerId;
    uint32_t        adapter;
    uint32_t        segmentId;
    uint32_t        nvmNamespace;
    size_t          warmups;
    size_t          repetitions;
    size_t          numQueues;
    size_t          queueDepth;
    size_t          numBlocks;
    size_t          startBlock;
    AccessPattern   pattern;
    const char*     filename;
    bool            write;
    bool            stats;
    uint32_t        domain;
    uint32_t        bus;
    uint32_t        devfn;
    std::string     cudaDeviceName;
    QueueLocation   queueLocation;
    bool            manager;

    Settings();
    void parseArguments(int argc, char** argv);

    std::string getDeviceBDF() const;
};

#endif
