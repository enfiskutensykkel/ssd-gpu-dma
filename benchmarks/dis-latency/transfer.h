#ifndef __DIS_LATENCY_TRANSFER_H__
#define __DIS_LATENCY_TRANSFER_H__

#include <memory>
#include <vector>
#include <cstddef>
#include <cstdint>
#include "segment.h"
#include "settings.h"
#include <nvm_types.h>


struct ChunkDescriptor
{
    SegmentPtr          prpList;
    DmaPtr              prpListMap;
    size_t              numBlocks;
    uint64_t            dptr1;
    uint64_t            dptr2;
};

typedef std::vector<ChunkDescriptor> ChunkList;


void setChunk(ChunkDescriptor& chunk, nvm_ctrl_t controller, DmaPtr target, const Settings& settings, uint32_t id, size_t size);


#endif
