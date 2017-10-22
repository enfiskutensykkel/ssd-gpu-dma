#ifndef __DIS_LATENCY_SETTINGS_H__
#define __DIS_LATENCY_SETTINGS_H__

#include <vector>
#include <cstddef>
#include <cstdint>


struct ManagerSettings
{
    int         cudaDevice;
    uint64_t    controllerId;
    uint32_t    ctrlAdapter;
    std::vector<uint32_t> adapters;

    ManagerSettings()
    {
        cudaDevice = -1;
        //controllerId = 0;
        controllerId = 0x10000;
        ctrlAdapter = 0;
    }
};


struct ClientSettings
{
    uint64_t    controllerId;
    uint32_t    ctrlAdapter;
    uint32_t    nvmNamespace;

    ClientSettings()
    {
        controllerId = 0x10000;
        ctrlAdapter = 0;
        nvmNamespace = 1;
    }
};

#endif
