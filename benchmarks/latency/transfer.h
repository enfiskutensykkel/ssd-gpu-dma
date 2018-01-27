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


/*
 * Returns number of buffer pages used
 */
size_t prepareRange(TransferList& list, const Controller& ctrl, size_t pageOffset, size_t startBlock, size_t numBlocks, bool write);


#endif
