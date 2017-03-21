#ifndef __NVME_QUEUE_H__
#define __NVME_QUEUE_H__
#ifdef __cplusplus
extern "C" {
#endif

#ifndef __CUDACC__
#define __host__
#define __device__
#endif


/* SQ doorbell register */
#define SQ_DBL(p, y, dstrd)    \
    ((volatile uint32_t*) (((volatile unsigned char*) (p)) + 0x1000 + ((2*(y)) * (4 << (dstrd)))) )


/* CQ doorbell register */
#define CQ_DBL(p, y, dstrd)    \
    ((volatile uint32_t*) (((volatile unsigned char*) (p)) + 0x1000 + ((2*(y) + 1) * (4 << (dstrd)))) )


#include "types.h"
struct completion;
struct command;


/* Enqueue a submission command
 * 
 * Enqueue a submission command in the specified SQ and return a pointer to
 * the queue slot in order to build the command inline in queue memory.
 *
 * Returns a pointer to the queue entry. or NULL if the queue is full.
 */
__host__ __device__
struct command* sq_enqueue(nvm_queue_t sq);


/* Poll completion queue
 *
 * Check the head of a completion queue for a new entry. The caller must
 * call dequeue manually.
 *
 * Returns a pointer to an unprocessed completion entry, or NULL if the queue
 * is empty.
 */
__host__ __device__
struct completion* cq_poll(nvm_queue_t cq);


/* Dequeue completion queue entry
 *
 * Dequeue a completion entry from the completion queue. If there is a ready
 * completion, this function will also update the SQ head pointer of the 
 * queue specified in the completion.
 *
 * Returns a pointer to the completion entry, or NULL if the queue is empty.
 */
__host__ __device__
struct completion* cq_dequeue(nvm_queue_t cq, nvm_controller_t ctrl);


/* Dequeue completion queue entry
 *
 * Dequeue a completion entry from the completion queue. If none are ready
 * at the time, this function will block until a controller timeout interval
 * or a ready completion. This function will also update the SQ head pointer
 * of the queue specified in the completion.
 *
 * Returns a pointer to the completion entry, or NULL if the queue is empty.
 *
 * Note that this can only be called from the host, as it makes no sense to
 * block a CUDA kernel.
 */
__host__
struct completion* cq_dequeue_block(nvm_queue_t cq, nvm_controller_t ctrl);


/* Update SQ tail pointer
 *
 * Submit all enqueued commands by ringing the doorbell.
 * The caller must make sure that all commands are prepared before calling
 * this.
 */
__host__ __device__
void sq_submit(nvm_queue_t sq);


/* Update controller's CQ head pointer
 *
 * Indicate that all completions are processed by ringing the doorbell.
 * All completion pointers acquired before this must be discarded after
 * calling this.
 */
__host__  __device__
void cq_update(nvm_queue_t cq); 


#ifdef __cplusplus
}
#endif
#endif
