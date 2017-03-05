#ifndef __NVME_H__
#define __NVME_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include "page.h"


/* Doorbell register */
#define SQ_DBL(p, y, dstrd)    \
    ((volatile void*) (((volatile unsigned char*) (p)) + 0x1000 + ((2*(y)) * (4 << (dstrd)))) )


#define CQ_DBL(p, y, dstrd)    \
    ((volatile void*) (((volatile unsigned char*) (p)) + 0x1000 + ((2*(y) + 1) * (4 << (dstrd)))) )


/* IO queue handle
 *
 * This structure represents an IO queue and holds information
 * about entries and size as well as doorbell handles.
 */
struct nvm_queue
{
    unsigned int        no;             // Queue number
    page_t              page;           // Page handle to where the queue is hosted
    size_t              max_entries;    // Maximum number of queue entries supported
    size_t              entry_size;     // Entry size
    uint32_t            head;           // Head pointer
    uint32_t            tail;           // Tail pointer
    int                 phase;          // Current phase bit
    volatile uint32_t*  db;             // Pointer to doorbell register
};

/* Convenience type for queue handle */
typedef struct nvm_queue* nvm_queue_t;


/* NVM controller handle
 *
 * This structure represents an NVMe controller and holds information
 * about controller properties and queues.
 *
 */
struct nvm_controller
{
    unsigned long   page_size;      // Memory page size
    uint8_t         dstrd;          // Doorbell stride (in encoded form)
    int             enabled;        // Controller enabled
    uint64_t        timeout;        // Controller timeout in milliseconds
    size_t          max_entries;    // Maximum queue entries supported
    size_t          cq_entry_size;  // CQ entry size
    size_t          sq_entry_size;  // SQ entry size
    int             max_queues;     // Maximum number of IO queues
    int             n_queues;       // Number queues
    nvm_queue_t*    queue_handles;  // Pointer to queues (even numbered queues=SQ, odd numbered=CQ)
};

/* Convenience type for controller handle */
typedef struct nvm_controller* nvm_controller_t;


#ifdef __cplusplus
}
#endif
#endif
