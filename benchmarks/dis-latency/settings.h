#ifndef __DIS_LATENCY_SETTINGS_H__
#define __DIS_LATENCY_SETTINGS_H__

#include <vector>
#include <cstddef>
#include <cstdint>
#include <nvm_command.h>


struct Settings
{
    int                 cudaDevice;
    uint64_t            ctrlId;
    uint32_t            ctrlAdapter;
    uint32_t            nvmNamespace;
    uint32_t            repetitions;
    size_t              blockSize;
    size_t              maxSize;
    std::vector<size_t> transferSizes;

    Settings()
    {
        cudaDevice = -1;
        //controllerId = 0;
        ctrlId = 0x10000;
        ctrlAdapter = 0;
        nvmNamespace = 1;
        repetitions = 10000;
        blockSize = 512;
        maxSize = 262144;

        transferSizes.push_back(1 << 9);
        transferSizes.push_back(1 << 10);
        transferSizes.push_back(1 << 11);
        transferSizes.push_back(1 << 12);
        transferSizes.push_back(1 << 13);
        transferSizes.push_back(1 << 14);
        transferSizes.push_back(1 << 15);
        transferSizes.push_back(1 << 16);
    }
};

#endif
