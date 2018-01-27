#include "ctrl.h"
#include "transfer.h"
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <nvm_util.h>
#include <cstdio>



size_t prepareRange(TransferList& list, const Controller& ctrl, size_t pageOff, size_t startBlock, size_t numBlocks, bool write)
{
    const size_t blockSize = ctrl.ns.lba_data_size;
    const size_t pageSize = ctrl.info.page_size;

    const size_t totalSize = numBlocks * blockSize;
    const size_t totalPages = NVM_PAGE_ALIGN(totalSize, pageSize) / pageSize;
    
    size_t page = 0;
    while (page < totalPages)
    {
        size_t transferPages = std::min(ctrl.info.max_data_pages, totalPages - page);
        
        Transfer t;
        t.write = write;
        t.startBlock = startBlock + NVM_PAGE_TO_BLOCK(pageSize, blockSize, page);
        t.numBlocks = std::min(numBlocks - t.startBlock, NVM_PAGE_TO_BLOCK(pageSize, blockSize, transferPages));
        t.startPage = pageOff + page;
        t.numPages = transferPages;

        list.push_back(t);
        page += transferPages;
    }

    return page;
}



//static void fillRandom(const Controller& ctrl, TransferList& list, size_t numBlocks, bool write)
//{
//    const size_t blockSize = ctrl.ns.lba_data_size;
//    const size_t pageSize = ctrl.ctrl->page_size;
//    size_t transferBlocks = NVM_PAGE_TO_BLOCK(pageSize, blockSize, ctrl.info.max_data_pages);
//
//    size_t startBlock = rand() % ctrl.ns.size;
//
//    while (numBlocks != 0)
//    {
//        transferBlocks = std::min(transferBlocks, numBlocks);
//
//        Transfer t;
//        t.write = write;
//        t.startBlock = startBlock;
//        t.numBlocks = transferBlocks;
//        t.pages = NVM_BLOCK_TO_PAGE(pageSize, blockSize, t.numBlocks);
//
//        list.push_back(t);
//
//        startBlock += transferBlocks;
//        numBlocks -= transferBlocks;
//    }
//}
//
