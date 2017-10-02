#include <nvm_util.h>
#include <nvm_types.h>
#include <nvm_command.h>
#include <algorithm>
#include <map>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include "transfer.h"
#include "dma.h"
#include "queue.h"
#include "settings.h"


static void createTuples(std::vector<TransferTuple>& tuples, 
                         size_t blockSize, 
                         size_t pageSize, 
                         size_t totalBlocks, 
                         size_t blocksPerChunk,
                         size_t startBlock,
                         size_t prpPages,
                         void* prpPtr,
                         const uint64_t* prpListAddrs,
                         const uint64_t* buffAddrs
                         )
{

    for (size_t blocks = 0, prpList = 0, buffPage = 0; blocks < totalBlocks; blocks += blocksPerChunk, ++prpList)
    {
        size_t chunkBlocks = std::min((totalBlocks - blocks), blocksPerChunk);
        size_t chunkSize = chunkBlocks * blockSize;

        TransferTuple tuple;
        tuple.pageIoAddr = buffAddrs[buffPage++];

        if (chunkSize <= pageSize)
        {
            tuple.prpListIoAddr = 0;
        }
        else if (chunkSize <= 2 * pageSize)
        {
            tuple.prpListIoAddr = buffAddrs[buffPage++];
        }
        else
        {
            if (prpPtr == nullptr || prpListAddrs == nullptr)
            {
                throw std::runtime_error("Invalid argument to createTuples");
            }

            size_t prpOffset = prpList * prpPages;
            tuple.prpListIoAddr = prpListAddrs[prpOffset];

            buffPage += nvm_prp_list(DMA_VADDR(prpPtr, pageSize, prpOffset), pageSize,
                    chunkSize - pageSize, &prpListAddrs[prpOffset], &buffAddrs[buffPage]);
        }

        tuple.startBlock = startBlock + blocks;
        tuple.numBlocks = chunkBlocks;

        tuples.push_back(tuple);
    }
}


static QueueTransferPtr createTransfer(nvm_ctrl_t ctrl, size_t totalBlocks, uint64_t startBlock, const DmaPtr buffer, const Settings& settings, nvm_queue_t* sq)
{
    const size_t totalSize = totalBlocks * settings.blockSize;
    const size_t blocksPerChunk = settings.chunkSize / settings.blockSize;
    const uint64_t* buffAddrs = (*buffer)->ioaddrs;

    // Initialize transfer descriptor
    QueueTransferPtr transfer(new QueueTransfer);
    transfer->queue = sq;
    transfer->totalSize = totalSize;
    transfer->chunkSize = settings.chunkSize;

    // Create PRP list (if necessary)
    const size_t prpListSize = nvm_prp_list_size((*buffer)->page_size, totalSize, settings.chunkSize);
    const size_t prpPages = nvm_prp_num_pages((*buffer)->page_size, settings.chunkSize);

    void* prpListVaddr = nullptr;
    const uint64_t* prpListAddrs = nullptr;

    if (prpListSize > 0)
    {
        DmaPtr prpLists = createHostBuffer(ctrl, prpListSize);
        transfer->prpLists = prpLists;
        prpListVaddr = (*prpLists)->vaddr;
        prpListAddrs = (*prpLists)->ioaddrs;
    }

    // Fill out PRP lists and create transfer tuples
    createTuples(transfer->transfers, settings.blockSize, (*buffer)->page_size, totalBlocks, blocksPerChunk,
            startBlock, prpPages, prpListVaddr, prpListAddrs, buffAddrs);

    return transfer;
}


void prepareTransfers(QueueTransferMap& transfers, nvm_ctrl_t ctrl, QueueList& queues, const DmaPtr buffer, const Settings& settings)
{
    const size_t totalBlocks = settings.numBlocks / (queues.size() - 1);
    uint64_t startBlock = settings.startBlock;

    for (size_t queueIdx = 1; queueIdx < queues.size(); ++queueIdx)
    {
        auto transfer = createTransfer(ctrl, totalBlocks, startBlock, buffer, settings, &queues[queueIdx]);
        transfers.insert(std::make_pair(transfer->queue->no, transfer));

        startBlock += totalBlocks;
    }

    const size_t remainingBlocks = settings.numBlocks - (startBlock - settings.startBlock);
    if (remainingBlocks > 0)
    {
        auto transfer = createTransfer(ctrl, remainingBlocks, startBlock, buffer, settings, &queues[1]);
        transfers.insert(std::make_pair(0, transfer));
    }
}

