#ifndef __SIMPLE_RDMA_SETTINGS_H__
#define __SIMPLE_RDMA_SETTINGS_H__

#include <cstddef>
#include <cstdint>


struct Settings
{
    int         cudaDevice;
    uint64_t    controllerId;
    uint32_t    nvmNamespace;
    size_t      numQueues;
    size_t      numBlocks;
    uint64_t    startBlock;
    size_t      repeatLoops;
    size_t      chunkSize;
    size_t      blockSize;
    bool        interleave;
    bool        write;
};


#endif
