#ifndef __NVME_CORE_H__
#define __NVME_CORE_H__
#ifdef __cplusplus
extern "C" {
#endif

#include "util.h"
#include "nvme.h"
#include <stdint.h>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif


// TODO: some kind of error lookup based on chapter 4.6.1.2.1


/* Pointer to command ID field (CID) in command */
#define COMMAND_ID(ptr) _REG(ptr, 2, 16)


/* Pointer to command ID field (CID) in completion */
#define COMPLETION_ID(ptr) _REG(ptr, 12, 16)


#define STATUS_DO_NOT_RETRY(ptr)        \
    !!((ptr)->dword[3] & (1 << 31))

#define STATUS_MORE(ptr)                \
    !!((ptr)->dword[3] & (1 << 30))

#define STATUS_CODE_TYPE(ptr)           \
    ((uint8_t) _RB((ptr)->dword[3], 27, 25))

#define STATUS_CODE(ptr)                \
    ((uint8_t) _RB((ptr)->dword[3], 24, 17))


/* NVM completion data type (16 bytes) */
struct completion
{
    uint32_t dword[4];
};


/* NVM command data type (64 bytes) */
struct command
{
    uint32_t dword[16];
};


/* Enqueue submission command
 * 
 * Get a free queue entry in a submission queue and increase the tail pointer.
 *
 * Returns a pointer to the queue entry. or NULL if the queue is full.
 */
__host__ __device__
struct command* sq_enqueue(nvm_queue_t sq);


/* Dequeue completion queue entry
 *
 * Dequeue a completion entry from the completion queue.
 *
 * Returns a pointer to the completion entry, or NULL if the queue is empty.
 */
__host__ __device__
struct completion* cq_dequeue(nvm_queue_t cq, nvm_controller_t controller);


/* Dequeue completion queue entry
 *
 * Dequeue a completion entry from the completion queue.
 * If none are available at the time, try again until a controller
 * timeout interval.
 *
 * Returns a pointer to the completion entry, or NULL if the queue is empty.
 */
__host__
struct completion* cq_dequeue_block(nvm_queue_t cq, nvm_controller_t controller);


/* Poll a completion queue
 *
 * Check the head of a completion queue for a new entry.
 *
 * Returns a pointer to an unprocessed completion entry, or NULL if the queue
 * is empty.
 */
__host__ __device__
struct completion* cq_poll(nvm_queue_t cq);


/* Update controller's SQ tail pointer
 *
 * Submit all outstanding commands by ringing the doorbell.
 * The caller must make sure that all commands are prepared before calling
 * this.
 */
__host__ 
void sq_submit(nvm_queue_t sq);


/* Update controller's CQ head pointer
 *
 * Indicate that all completions are processed by ringing the doorbell.
 * All completion pointers acquired before this must be discarded after
 * calling this.
 */
__host__ 
void cq_update(nvm_queue_t cq); 


// TODO: need some function to submit from kernel

#ifdef __cplusplus
}
#endif
#endif
