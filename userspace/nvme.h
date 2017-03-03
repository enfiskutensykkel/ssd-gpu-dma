#ifndef __NVME_H__
#define __NVME_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>


/* Convenience function for creating a bit mask */
static inline uint64_t bitmask(int hi, int lo)
{
    uint64_t mask = 0;

    for (int i = lo; i <= hi; ++i)
    {
        mask |= 1UL << i;
    }

    return mask;
}


/* Extract specific bits */
#define _RB(v, hi, lo)   \
    ( ( (v) & bitmask((hi), (lo)) ) >> (lo) )


/* Set specifics bits */
#define _WB(v, hi, lo)   \
    ( ( (v) << (lo) ) & bitmask((hi), (lo)) )


/* Offset to a register */
#define _REG(p, offs, bits) \
    ((volatile uint##bits##_t *) (((volatile unsigned char*) (p)) + (offs)))


/* Controller registers */
#define CAP(p)          _REG(p, 0x0000, 64)     // Controller Capabilities
#define CC(p)           _REG(p, 0x0014, 32)     // Controller Configuration
#define CSTS(p)         _REG(p, 0x001c, 32)     // Controller Status
#define AQA(p)          _REG(p, 0x0024, 32)     // Admin Queue Attributes
#define ASQ(p)          _REG(p, 0x0028, 64)     // Admin Submission Queue Base Address
#define ACQ(p)          _REG(p, 0x0030, 64)     // Admin Completion Queue Base Address


/* Read bit fields */
#define CAP$MPSMAX(p)   _RB(*CAP(p), 55, 52)    // Memory Page Size Maximum
#define CAP$MPSMIN(p)   _RB(*CAP(p), 51, 48)    // Memory Page Size Minimum
#define CAP$DSTRD(p)    _RB(*CAP(p), 35, 32)    // Doorbell Stride
#define CAP$TO(p)       _RB(*CAP(p), 31, 24)    // Timeout
#define CAP$CQR(p)      _RB(*CAP(p), 16, 16)    // Contiguous Queues Required
#define CAP$MQES(p)     _RB(*CAP(p), 15,  0)    // Maximum Queue Entries Supported

#define CSTS$RDY(p)     _RB(*CSTS(p), 0,  0)    // Ready


/* Write bit fields */
#define CC$IOCQES(v)    _WB(v, 23, 20)          // IO Completion Queue Entry Size
#define CC$IOSQES(v)    _WB(v, 19, 16)          // IO Submission Queue Entry Size
#define CC$MPS(v)       _WB(v, 10,  7)          // Memory Page Size
#define CC$CSS(v)       _WB(0,  3,  1)          // IO Command Set Selected (0=NVM Command Set)
#define CC$EN(v)        _WB(v,  0,  0)          // Enable


/* Doorbell register */
#define SQ_DBL(p, y, dstrd)    \
    ((volatile void*) ((volatile unsigned char*) (p)) + 0x1000 + ((2*(y)) * (4 << (dstrd))) )


#define CQ_DBL(p, y, dstrd)    \
    ((volatile void*) ((volatile unsigned char*) (p)) + 0x1000 + ((2*(y) + 1) * (4 << (dstrd))) )




/* Forward declaration of queue handle */
struct nvm_queue;
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
    size_t          max_queues;     // Maximum number of IO queues
    size_t          n_queues;       // Number queues
    nvm_queue_t*    queue_handles;  // Pointer to queues (even numbered queues=SQ, odd numbered=CQ)
};

/* Convenience type for controller handle */
typedef struct nvm_controller* nvm_controller_t;



/* IO queue handle
 *
 * This structure represents an IO queue and holds information
 * about entries and size as well as doorbell handles.
 */
struct nvm_queue
{
    unsigned int    no;             // Queue number
    int             dev;            // CUDA device queue is hosted on (-1 for no device)
    void*           ptr;            // Pointer to queue memory
    size_t          max_entries;    // Maximum number of queue entries supported
    size_t          entry_size;     // Entry size
    size_t          head;           // Head pointer
    size_t          tail;           // Tail pointer
    int             phase;          // Current phase bit
    volatile void*  db;             // Pointer to doorbell register
};



#ifdef __cplusplus
}
#endif
#endif
