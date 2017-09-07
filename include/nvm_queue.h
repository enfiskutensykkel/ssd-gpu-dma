#ifndef __DIS_NVM_QUEUE_H__
#define __DIS_NVM_QUEUE_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

struct nvm_controller;
struct nvm_queue;
struct nvm_command;
struct nvm_completion;


/*
 * Clear a queue handle.
 *
 * Clear a queue handle struct. The caller must clear the queue memory 
 * manually before using the handle.
 *
 * Note: vaddr must be page-aligned
 */
void nvm_queue_clear(struct nvm_queue* q, const struct nvm_controller* ctrl, int cq, uint16_t no, void* vaddr, uint64_t ioaddr);



/* 
 * Enqueue a submission command.
 * 
 * Enqueue a submission command in the specified SQ and return a pointer to
 * the queue slot in order to build the command inline in queue memory.
 *
 * Returns a pointer to the queue entry. or NULL if the queue is full.
 */
struct nvm_command* sq_enqueue(struct nvm_queue* sq);


/* 
 * Poll completion queue.
 *
 * Check the head of a completion queue for a new entry. The caller must
 * call dequeue manually.
 *
 * Returns a pointer to an unprocessed completion entry, or NULL if the queue
 * is empty.
 */
struct nvm_completion* cq_poll(const struct nvm_queue* cq);


/* 
 * Dequeue completion queue entry.
 *
 * Dequeue a completion entry from the completion queue. If there is no ready
 * completions, this function returns NULL.
 *
 * The caller must update the corresponding SQ manually.
 *
 * Returns a pointer to the completion entry, or NULL if the queue is empty.
 */
struct nvm_completion* cq_dequeue(struct nvm_queue* cq);


/* 
 * Dequeue completion queue entry.
 *
 * Dequeue a completion entry from the completion queue. If none are ready
 * at the time, this function will block until a controller timeout interval
 * or a ready completion. 
 *
 * Returns a pointer to the completion entry, or NULL if the queue is empty or
 * on timeout.
 */
struct nvm_completion* cq_dequeue_block(struct nvm_queue* cq, uint64_t timeout);


/* 
 * Update SQ tail pointer.
 *
 * Submit all enqueued commands by ringing the doorbell.
 * The caller must make sure that all commands are prepared before calling
 * this.
 */
void sq_submit(const struct nvm_queue* sq);


/* 
 * Update SQ head pointer.
 *
 * Update SQ head pointer according to the head pointer field in a completion.
 */
int sq_update(struct nvm_queue* sq, const struct nvm_completion* cpl);


/* 
 * Update controller's CQ head pointer.
 *
 * Indicate that all completions are processed by ringing the doorbell.
 * All completion pointers acquired before this must be discarded after
 * calling this.
 */
void cq_update(const struct nvm_queue* cq); 


#ifdef __cplusplus
}
#endif
#endif
