#ifndef __SIMPLE_RDMA_BENCHMARK_H__
#define __SIMPLE_RDMA_BENCHMARK_H__

#include <nvm_types.h>
#include <memory>
#include <vector>
#include <cstddef>
#include <cstdint>
#include "manager.h"
#include "queue.h"


typedef std::shared_ptr<nvm_dma_t> DmaPtr;

/*
 * Benchmark configuration
 */
struct BenchmarkSettings
{
    int             dev_id;         // CUDA device ID
    uint64_t        ctrl_id;        // NVMe controller ID
    uint32_t        ns;             // NVMe namespace
    uint16_t        n_queues;       // Number of queues to use
    size_t          n_blks;         // Number of blocks
    size_t          blk_start;      // LBA start block
    size_t          blk_size;       // Block size
    size_t          page_size;      // Page size
    size_t          total_size;     // Total size to be transferred
    size_t          chunk_size;     // How much can be transferred at once
    size_t          n_prps;         // How many PRPs are needed to describe a chunk
    size_t          n_prp_pages;    // How many PRP pages needed to describe a chunk
    size_t          n_prp_lists;    // How many PRP lists are needed to describe the entire transfer

};


struct __align__(16) TransferTuple
{
    uint64_t page_addr;
    uint64_t prp_list_addr;
    uint64_t blk_start;
    uint16_t n_blks;
};


typedef std::vector<TransferTuple> TupleList;



void prepare_transfer_tuples(TupleList& tuples, DmaPtr prps, DmaPtr buff, const BenchmarkSettings& settings);


uint64_t time_transfer(TupleList& tuples, 
                       nvm_ctrl_t ctrl, 
                       QueueManagerPtr queues, 
                       uint32_t ns, 
                       void* vaddr, 
                       void* devptr, 
                       size_t size, 
                       int device);

#endif
