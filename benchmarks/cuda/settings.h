#ifndef __SETTINGS_H__
#define __SETTINGS_H__

#include <string>
#include <cstddef>
#include <cstdint>


struct Settings
{
    uint32_t        cudaDevice;
    uint64_t        cudaDeviceId;
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
    uint32_t        domain;
    uint32_t        bus;
    uint32_t        devfn;

    Settings();
    void parseArguments(int argc, char** argv);

    static std::string usageString(const std::string& name);

    std::string getDeviceBDF() const;
};

#endif
