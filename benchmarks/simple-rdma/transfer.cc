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


void controlTransferMemory(const TransferList& list, const DmaPtr buffer)
{
    std::map<uint64_t, int64_t> addrCounts;
    for (size_t i = 0; i < (*buffer)->n_ioaddrs; ++i)
    {
        addrCounts[(*buffer)->ioaddrs[i]] = 1; // We initialise it to one
    }

    for (const TransferPtr& transfer: list)
    {
        for (size_t chunkNo = 0; chunkNo < transfer->chunks.size(); ++chunkNo)
        {
            const Chunk& chunk = transfer->chunks[chunkNo];

            size_t prpOffset = nvm_prp_num_pages(transfer->pageSize, transfer->chunkSize) * chunkNo;
            size_t chunkSize = chunk.numBlocks * transfer->blockSize;
            
            --addrCounts[chunk.pageIoAddr];

            if (chunkSize <= transfer->pageSize)
            {
                continue;
            }
            else if (chunkSize <= 2 * transfer->pageSize)
            {
                --addrCounts[chunk.prpListIoAddr];
                continue;
            }

            const uint64_t* prpPtr = (const uint64_t*) DMA_WND_VADDR(*transfer->prpList, prpOffset);
            for (size_t i = 0; i < (chunkSize / transfer->pageSize) - 1; ++i)
            {
                --addrCounts[prpPtr[i]];
            }
        }
    }

    for (const auto& addr: addrCounts)
    {
        if (addr.second != 0)
        {
            std::fprintf(stderr, "%lx %zu\n", addr.first, addr.second);
            throw std::runtime_error("Memory pages are not contiguous");
        }
    }
}


constexpr static size_t transferPages(size_t numBlocks, size_t blockSize, size_t pageSize)
{
    return (numBlocks * blockSize) / pageSize;
}


constexpr static size_t blocksPerChunk(const Settings& settings)
{
    return settings.chunkSize / settings.blockSize + (settings.chunkSize % settings.blockSize != 0);
}


static size_t setChunk(Chunk& chunk, const TransferPtr& transfer, void* prpPtr, const uint64_t* prpPages, const uint64_t* bufferPages, uint64_t startBlock, uint64_t numBlocks)
{
    chunk.pageIoAddr = bufferPages[0];
    chunk.prpListIoAddr = 0;
    chunk.startBlock = startBlock;
    chunk.numBlocks = numBlocks;

    size_t chunkSize = numBlocks * transfer->blockSize;

    if (chunkSize <= transfer->pageSize)
    {
        return 1;
    }
    else if (chunkSize <= 2 * transfer->pageSize)
    {
        chunk.prpListIoAddr = bufferPages[1];
        return 2;
    }

    size_t bufferPageCount = nvm_prp_list(prpPtr, transfer->pageSize, chunkSize - transfer->pageSize, prpPages, &bufferPages[1]);
    chunk.prpListIoAddr = prpPages[0];

    return 1 + bufferPageCount;
}


void prepareTransfers(TransferList& list, nvm_ctrl_t ctrl, QueueList& queues, const DmaPtr buffer, const Settings& settings)
{
    if (transferPages(settings.numBlocks, settings.blockSize, (*buffer)->page_size) > (*buffer)->n_ioaddrs)
    {
        throw std::runtime_error("Transfer size is greater than buffer size");
    }

    list.clear();
    for (auto queueIt = queues.begin() + 1; queueIt != queues.end(); ++queueIt)
    {
        size_t transferSize = settings.numBlocks * settings.blockSize;

        TransferPtr transfer(new Transfer);
        transfer->queue = &*queueIt;
        transfer->prpList = createHostBuffer(ctrl, nvm_prp_list_size((*buffer)->page_size, transferSize, settings.chunkSize));
        transfer->nvmNamespace = settings.nvmNamespace;
        transfer->pageSize = (*buffer)->page_size;
        transfer->blockSize = settings.blockSize;
        transfer->chunkSize = settings.chunkSize;

        list.push_back(transfer);
    }

    const uint64_t* bufferPages = (*buffer)->ioaddrs;
    size_t bufferPage = 0;

    uint64_t startBlock = settings.startBlock;
    uint64_t remainingBlocks = settings.numBlocks;

    auto first = list.begin();
    auto last = list.end();
    auto transferIt = first;

    while (remainingBlocks > 0)
    {
        const size_t numBlocks = std::min(remainingBlocks, blocksPerChunk(settings));
        
        TransferPtr& transfer = *transferIt;

        const size_t prpPage = transfer->chunks.size() * nvm_prp_num_pages(transfer->pageSize, transfer->chunkSize);
        const uint64_t* prpPages = &(*transfer->prpList)->ioaddrs[prpPage];
        void* prpPtr = DMA_WND_VADDR(*transfer->prpList, prpPage);

        Chunk chunk;
        const size_t numPages = setChunk(chunk, transfer, prpPtr, prpPages, &bufferPages[bufferPage], startBlock, numBlocks);
        transfer->chunks.push_back(chunk);

        bufferPage += numPages;
        startBlock += numBlocks;
        remainingBlocks -= numBlocks;

        if (++transferIt == last)
        {
            transferIt = first;
        }
    }
}

