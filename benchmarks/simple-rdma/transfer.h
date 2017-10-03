#ifndef __SIMPLE_RDMA_TRANSFER_H__
#define __SIMPLE_RDMA_TRANSFER_H__

#include <nvm_types.h>
#include <vector>
#include <memory>
#include <cstddef>
#include <cstdint>
#include "dma.h"
#include "queue.h"
#include "settings.h"

#ifndef __CUDACC__
#define __host__
#endif


struct Chunk
{
    uint64_t        pageIoAddr;
    uint64_t        prpListIoAddr;
    uint64_t        startBlock;
    uint16_t        numBlocks;
};


struct Transfer
{
    nvm_queue_t*        queue;
    DmaPtr              prpList;
    uint32_t            nvmNamespace;
    size_t              pageSize;
    size_t              blockSize;
    size_t              chunkSize;
    std::vector<Chunk>  chunks;
};


typedef std::shared_ptr<Transfer> TransferPtr;

typedef std::vector<TransferPtr> TransferList;


__host__ void prepareTransfers(TransferList& transfers,
                               nvm_ctrl_t controller,
                               QueueList& queues, 
                               const DmaPtr buffer, 
                               const Settings& settings);

#endif
