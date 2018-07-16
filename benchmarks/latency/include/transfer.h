#ifndef __LATENCY_BENCHMARK_TRANSFER_H__
#define __LATENCY_BENCHMARK_TRANSFER_H__

#include "ctrl.h"
#include "queue.h"
#include "buffer.h"
#include <map>
#include <vector>
#include <memory>
#include <cstddef>
#include <cstdint>


/* 
 * Describe a transfer of "chunk" -- the maximum transfer of a single command.
 */
struct Chunk
{
    size_t      startBlock;
    size_t      numBlocks;
    size_t      startPage;
    size_t      numPages;
};



/* Convenience types */
typedef std::vector<Chunk> ChunkList;
typedef ChunkList::const_iterator ChunkPtr;



/*
 * Describes a total transfer.
 */
struct Transfer
{
    QueuePtr    queue;
    MemoryBufferPtr buffer;
    size_t      count;
    size_t      offset;
    size_t      pageOffset;
    ChunkList   chunks;
};


typedef std::shared_ptr<Transfer> TransferPtr;
typedef std::map<uint16_t, TransferPtr> TransferMap;



#endif /* __LATENCY_BENCHMARK_SETTINGS_H__ */
