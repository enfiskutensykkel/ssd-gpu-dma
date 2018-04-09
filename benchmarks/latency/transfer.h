#ifndef __TRANSFER_H__
#define __TRANSFER_H__

#include "ctrl.h"
#include <vector>
#include <cstddef>
#include <cstdint>
#include <vector>


struct Transfer
{
    bool        write;
    size_t      startBlock;
    size_t      numBlocks;
    size_t      startPage;
    size_t      numPages;
};


typedef std::vector<Transfer> TransferList;
typedef TransferList::const_iterator TransferPtr;


size_t prepareRange(TransferList& list, const Controller& ctrl, bool write, size_t pageOffset, uint64_t start, size_t numBlocks);


size_t fillRandom(TransferList& list, const Controller& ctrl, bool write, size_t numBlocks, bool perChunk, bool perPage);


#endif
