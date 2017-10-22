#include <nvm_util.h>
#include <nvm_types.h>
#include <nvm_command.h>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include "transfer.h"
#include "segment.h"

using std::runtime_error;


void setDataPointer(nvm_cmd_t* cmd, DmaPtr target, DmaPtr prpList, size_t blockSize, size_t chunkSize)
{
    if (chunkSize > (*target)->n_ioaddrs * (*target)->page_size)
    {
        throw runtime_error("Invalid argument");
    }

    size_t numTransferPages = (chunkSize / (*target)->page_size) + (chunkSize % (*target)->page_size != 0);

    //size_t startPage = (*target)->n_ioaddrs - numTransferPages;
    size_t startPage = 0;
    nvm_cmd_dptr(cmd, (*target)->ioaddrs[startPage], 0);

    if (chunkSize <= (*target)->page_size)
    {
        return;
    }
    else if (chunkSize <= 2 * (*target)->page_size)
    {
        nvm_cmd_dptr(cmd, (*target)->ioaddrs[startPage], (*target)->ioaddrs[startPage + 1]);
    }

    size_t numPrpPages = nvm_prp_num_pages((*target)->page_size, chunkSize);
    if (numPrpPages > (*prpList)->n_ioaddrs)
    {
        throw runtime_error("Invalid argument");
    }

    //size_t startPrp = (*prpList)->n_ioaddrs - numPrpPages;
    size_t startPrp = 0;

    nvm_prp_list((*prpList)->vaddr, (*target)->page_size, chunkSize - (*target)->page_size,
            &(*prpList)->ioaddrs[startPrp], &(*target)->ioaddrs[startPage + 1]);

    nvm_cmd_dptr(cmd, (*target)->ioaddrs[startPage], (*prpList)->ioaddrs[startPrp]);
}





