#ifndef __SETTINGS_H__
#define __SETTINGS_H__

#include <cstddef>
#include <cstdint>


struct Settings
{
    int             cudaDevice;
    const char*     blockDevicePath;
    const char*     controllerPath;
    uint32_t        controllerId;
    uint32_t        adapter;
    uint32_t        segmentId;
    uint32_t        nvmNamespace;
    size_t          numPages;
    size_t          startBlock;
    bool            stats;
    const char*     verify;
    uint32_t        numThreads;

    Settings();
    void parseArguments(int argc, char** argv);
};

#endif
