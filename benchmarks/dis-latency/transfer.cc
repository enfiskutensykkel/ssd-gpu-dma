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
#include "settings.h"


void setChunk(ChunkDescriptor& chunk, nvm_ctrl_t ctrl, DmaPtr target, const Settings& settings, uint32_t id, size_t size)
{
    if (size > (*target)->n_ioaddrs * (*target)->page_size)
    {
        throw std::runtime_error("Invalid argument");
    }

    chunk.dptr1 = (*target)->ioaddrs[0];
    chunk.dptr2 = 0;
    chunk.numBlocks = size / settings.blockSize;

    if (size <= 2 * (*target)->page_size)
    {
        chunk.dptr2 = (*target)->ioaddrs[1];
        return;
    }

    

    chunk.prpList = createSegment(id, 0x1000);
    chunk.prpListMap = createDmaMapping(chunk.prpList, ctrl, settings.ctrlAdapter);

    const uint64_t* bufferPages = (*target)->ioaddrs;
    chunk.dptr2 = (*chunk.prpListMap)->ioaddrs[0];

    nvm_prp_list((*chunk.prpListMap)->vaddr, ctrl->page_size, size - ctrl->page_size, (*chunk.prpListMap)->ioaddrs, bufferPages);
}
