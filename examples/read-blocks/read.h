#ifndef __LIBNVM_SAMPLES_READ_BLOCKS_READ_H__
#define __LIBNVM_SAMPLES_READ_BLOCKS_READ_H__

#include <stdint.h>
#include <stdbool.h>
#include <nvm_types.h>
#include "args.h"


/*
 * Information about controller and namespace.
 */
struct disk_info
{
    uint32_t    ns_id;
    size_t      max_data_size;
    size_t      page_size;
    size_t      block_size;
};



struct queue_pair
{
    nvm_dma_t*  sq_mem;
    nvm_dma_t*  cq_mem;
    nvm_queue_t sq;
    nvm_queue_t cq;
    bool        stop;
    size_t      num_cpls;
};



int get_disk_info(nvm_aq_ref ref, struct disk_info* info, uint32_t ns_id, void* ptr, uint64_t ioaddr, bool show);


int create_queue_pair(nvm_aq_ref ref, struct queue_pair* qp, nvm_dma_t* cq_mem, nvm_dma_t* sq_mem);


int read_and_dump(const struct disk_info* disk, struct queue_pair* qp, const nvm_dma_t* buffer, const struct options* args);


int write_blocks(const struct disk_info* disk, struct queue_pair* qp, const nvm_dma_t* buffer, const struct options* args);


#endif
