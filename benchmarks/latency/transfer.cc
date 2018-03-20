#include "ctrl.h"
#include "transfer.h"
#include <vector>
#include <algorithm>
#include <nvm_util.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>



size_t prepareRange(TransferList& list, const Controller& ctrl, bool write, size_t pageOffset, uint64_t start, size_t numBlocks)
{
    const size_t blockSize = ctrl.ns.lba_data_size;
    const size_t pageSize = ctrl.info.page_size;

    size_t pages = 0;
    uint64_t currBlock = 0;
    const size_t numBlocksPerTransfer = NVM_PAGE_TO_BLOCK(pageSize, blockSize, ctrl.info.max_data_size / pageSize);

    while (currBlock != numBlocks)
    {
        size_t transferBlocks = std::min(numBlocks - currBlock, numBlocksPerTransfer);
        size_t transferPages = NVM_PAGE_ALIGN(transferBlocks * blockSize, pageSize) / pageSize;
        size_t currPage = NVM_BLOCK_TO_PAGE(pageSize, blockSize, currBlock);

        Transfer t;
        t.write = write;
        t.startBlock = start + currBlock;
        t.numBlocks = transferBlocks;
        t.startPage = pageOffset + currPage;
        t.numPages = transferPages;

        list.push_back(t);
        currBlock += transferBlocks;
        pages += transferPages;
    }

    return pages;
}


size_t fillRandom(TransferList& list, const Controller& ctrl, bool write, size_t numBlocks)
{
    const size_t blockSize = ctrl.ns.lba_data_size;
    const size_t pageSize = ctrl.info.page_size;
    const size_t maxBlock = ctrl.ns.size / blockSize;
    const size_t chunkSize = ctrl.info.max_data_size;

    size_t pages = 0;
    uint64_t startBlock = rand() % maxBlock;

    for (size_t block = 0; block < numBlocks; block += std::min(chunkSize, numBlocks - block))
    {
        Transfer t;
        t.write = write;
        t.startBlock = startBlock + block;
        t.numBlocks = std::min(chunkSize, numBlocks - block);
        t.startPage = pages;
        t.numPages = NVM_PAGE_ALIGN(t.numBlocks * blockSize, pageSize) / pageSize;

        list.push_back(t);
        pages += t.numPages;
    }

    return pages;
}


//size_t fillRandom(TransferList& list, const Controller& ctrl, bool write, size_t numBlocks)
//{
//    const size_t blockSize = ctrl.ns.lba_data_size;
//    const size_t pageSize = ctrl.info.page_size;
//    const size_t maxBlock = ctrl.ns.size / blockSize;
//
//    size_t pages = 0;
//
//    numBlocks = NVM_PAGE_ALIGN(numBlocks * blockSize, pageSize) / blockSize;
//    for (size_t block = 0; block < numBlocks; block += NVM_PAGE_TO_BLOCK(pageSize, blockSize, 1))
//    {
//        Transfer t;
//        t.write = write;
//        t.startBlock = rand() % maxBlock;
//        t.numBlocks = NVM_PAGE_TO_BLOCK(pageSize, blockSize, 1);
//        t.startPage = pages;
//        t.numPages = 1;
//
//        list.push_back(t);
//        ++pages;
//    }
//
//    return pages;
//}




