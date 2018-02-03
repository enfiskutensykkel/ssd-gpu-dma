#ifndef __SETTINGS_H__
#define __SETTINGS_H__

#include <cstddef>
#include <cstdint>


enum AccessPattern : int
{
    LINEAR,              // All threads read the same sequential chunk
    SEQUENTIAL,          // All threads read the same (larger) sequential chunk with different offsets
    RANDOM               // All threads read random blocks
};


struct Settings
{
    int             cudaDevice;
    const char*     controllerPath;
    uint32_t        controllerId;
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
    bool            remote;

    Settings();
    void parseArguments(int argc, char** argv);
};

#endif
