#ifndef __BENCHMARK_QUEUEPAIR_H__
#define __BENCHMARK_QUEUEPAIR_H__

#include <nvm_types.h>
#include <cstdint>
#include "buffer.h"
#include "settings.h"
#include "ctrl.h"


struct __align__(64) QueuePair
{
    uint32_t            pageSize;
    uint32_t            blockSize;
    uint32_t            nvmNamespace;
    uint32_t            pagesPerChunk;
    bool                doubleBuffered;
    void*               prpList;
    uint64_t            prpListIoAddr;
    nvm_queue_t         sq;
    nvm_queue_t         cq;
};


__host__ DmaPtr prepareQueuePair(QueuePair& qp, const Controller& ctrl, const Settings& settings);

#endif
