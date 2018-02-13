#ifndef __SETTINGS_H__
#define __SETTINGS_H__

#include <cstddef>
#include <cstdint>


struct Settings
{
    uint32_t        cudaDevice;
    const char*     blockDevicePath;
    const char*     controllerPath;
    uint64_t        controllerId;
    uint32_t        adapter;
    uint32_t        segmentId;
    uint32_t        nvmNamespace;
    bool            doubleBuffered;
    size_t          numChunks;
    size_t          numPages;
    size_t          startBlock;
    bool            stats;
    const char*     output;
    size_t          numThreads;

    Settings();
    void parseArguments(int argc, char** argv);
};

#endif
