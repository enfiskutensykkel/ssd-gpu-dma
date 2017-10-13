#ifndef __DIS_LATENCY_SETTINGS_H__
#define __DIS_LATENCY_SETTINGS_H__

#include <cstddef>
#include <cstdint>


struct Settings
{
    int         cudaDevice;
    uint64_t    controllerId;
    uint32_t    disAdapter;
    uint32_t    nvmNamespace;

    Settings();
};

#endif
