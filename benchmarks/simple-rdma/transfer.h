#ifndef __SIMPLE_RDMA_TRANSFER_H__
#define __SIMPLE_RDMA_TRANSFER_H__

#include <nvm_types.h>
#include <map>
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


struct TransferTuple
{
    uint64_t        pageIoAddr;
    uint64_t        prpListIoAddr;
    uint64_t        startBlock;
    uint16_t        numBlocks;
};

struct QueueTransfer
{
    nvm_queue_t*                queue;
    DmaPtr                      prpLists;
    size_t                      totalSize;
    size_t                      chunkSize;
    std::vector<TransferTuple>  transfers;
};

typedef std::shared_ptr<QueueTransfer> QueueTransferPtr;


typedef std::map<uint16_t, QueueTransferPtr> QueueTransferMap;


__host__
void prepareTransfers(QueueTransferMap& transfers,
                      nvm_ctrl_t controller,
                      QueueList& queues, 
                      const DmaPtr buffer, 
                      const Settings& settings);

#endif
