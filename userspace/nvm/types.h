#ifndef __NVM_TYPES_H__
#define __NVM_TYPES_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include "memory.h"


/* SQ doorbell register offset */
#define SQ_DBL(p, y, dstrd)    \
    ((volatile uint32_t*) (((volatile unsigned char*) (p)) + 0x1000 + ((2*(y)) * (4 << (dstrd)))) )


/* CQ doorbell register offset */
#define CQ_DBL(p, y, dstrd)    \
    ((volatile uint32_t*) (((volatile unsigned char*) (p)) + 0x1000 + ((2*(y) + 1) * (4 << (dstrd)))) )


/* IO queue handle
 *
 * This structure represents an IO queue and holds information about
 * queue entries and size as well as a pointer to the doorbell register.
 *
 * Note: the queue handle does not "own" the memory buffer it points to.
 * Memory management must be handled elsewhere.
 */
struct nvm_queue
{
    uint16_t            no;             // Queue number (must be unique per SQ/CQ pair)
    size_t              max_entries;    // Maximum number of queue entries supported
    size_t              entry_size;     // The size of a queue entry
    uint32_t            head;           // Head pointer
    uint32_t            tail;           // Tail pointer
    int                 phase;          // Current phase bit
    void*               virt_addr;      // Virtual address of the queue (may be device memory)
    uint64_t            bus_addr;       // Physical/bus address of the queue (only 1 page per queue supported)
    volatile uint32_t*  db;             // Pointer to doorbell register (write only)
};


/* Convenience type for queue handle */
typedef struct nvm_queue nvm_queue_t;


/* NVM controller handle
 *
 * This structure represents an NVMe controller and holds information
 * about controller properties and queues.
 */
struct nvm_controller
{
    size_t          page_size;      // Memory page size (MPS) used by the controller
    uint8_t         dstrd;          // Doorbell stride (in encoded form)
    uint64_t        timeout;        // Controller timeout in milliseconds
    uint16_t        max_queues;     // Maximum number of IO queue pairs
    size_t          max_out_cmds;   // Maximum outstanding commands (MAXCMD)
    size_t          max_data_size;  // Maximum data transfer size (MDTS)
    size_t          max_entries;    // Maximum queue entries supported
    size_t          cq_entry_size;  // CQ entry size (CQES)
    nvm_queue_t     admin_cq;       // Admin completion queue (ACQ)
    page_t          admin_cq_page;  // ACQ page descriptor
    size_t          sq_entry_size;  // SQ entry size (SQES)
    nvm_queue_t     admin_sq;       // Admin submission queue (ASQ)
    page_t          admin_sq_page;  // ASQ page descriptor
    volatile void*  reg_ptr;        // Pointer to registers and doorbell registers
    page_t          identify;       // Identify controller structure
    uint32_t        n_ns;           // Number of namespaces
};


/* Convenience type for controller handle */
typedef struct nvm_controller nvm_ctrl_t;


#ifdef __cplusplus
}
#endif
#endif
