#ifndef __LIBNVM_SAMPLES_INTEGRITY_H__
#define __LIBNVM_SAMPLES_INTEGRITY_H__

#include <nvm_types.h>
#include <stdio.h>
#include <stdint.h>


/* Memory descriptor */
struct buffer
{
    uint32_t                id;
    uint32_t                adapter;
    void*                   buffer;
    nvm_dma_t*              dma;
};


/* Queue descriptor */
struct queue
{
    struct buffer           qmem;
    nvm_queue_t             queue;
    size_t                  counter;
};


/* Disk descriptor */
struct disk
{
    size_t      page_size;
    size_t      max_data_size;
    uint32_t    ns_id;
    size_t      block_size;
};


int create_buffer(struct buffer* b, nvm_aq_ref, size_t size, uint32_t adapter, uint32_t id);


void remove_buffer(struct buffer* b);



int create_queue(struct queue* q, nvm_aq_ref ref, const struct queue* cq, uint16_t qno, uint32_t adapter, uint32_t id);


void remove_queue(struct queue* q);



int disk_write(const struct disk* disk, struct buffer* buffer, struct queue* queues, uint16_t n_queues, FILE* fp, off_t size);

int disk_read(const struct disk* disk, struct buffer* buffer, struct queue* queues, uint16_t n_queues, FILE* fp, off_t size);


#endif
