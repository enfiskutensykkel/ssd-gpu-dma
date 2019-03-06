#ifndef __NVM_QUEUE_H__
#define __NVM_QUEUE_H__

#ifndef __CUDACC__
#define __device__
#define __host__
#endif

#include <nvm_util.h>
#include <nvm_types.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>


/*
 * Clear queue descriptor.
 *
 * Initialize an empty queue descriptor. 
 * The user must clear the queue memory manually before using the handle.
 *
 * Note: vaddr must be page-aligned and at least one page.
 */
#ifdef __cplusplus
extern "C" {
#endif
__host__
void nvm_queue_clear(nvm_queue_t* q,            // NVM queue descriptor
                     const nvm_ctrl_t* ctrl,    // NVM controller handle
                     bool cq,                   // Is this a completion queue or submission queue?
                     uint16_t no,               // Queue number
                     uint16_t qs,               // Queue size (number of entries)
                     bool local,                // Is this local or remote memory
                     volatile void* vaddr,      // Virtual address to queue memory
                     uint64_t ioaddr);          // Bus address to queue memory (as seen from the controller)
#ifdef __cplusplus
}
#endif


/*
 * Reset queue descriptor and set all members to initial state.
 *
 * Note: this function should not be used if the queue has been created but
 *       not yet deleted, as it will lead to inconsistent state for the
 *       controller.
 */
#ifdef __cplusplus
extern "C" {
#endif
__host__
void nvm_queue_reset(nvm_queue_t* q);
#ifdef __cplusplus
}
#endif



/* 
 * Enqueue a submission command.
 * 
 * Enqueue a submission command in the specified SQ and return a pointer to
 * the queue slot in order to build the command inline in queue memory.
 *
 * Returns a pointer to the queue entry. or NULL if the queue is full.
 */
__host__ __device__ static inline
nvm_cmd_t* nvm_sq_enqueue(nvm_queue_t* sq)
{
    // Check if queue has available slots
    if (((uint16_t) (sq->tail - sq->head)) < sq->qs)
    {
        // Take slot and end of queue
        nvm_cmd_t* cmd = (nvm_cmd_t*) (((unsigned char*) sq->vaddr) + sq->es * (sq->tail % sq->qs));

        // Increase tail pointer and invert phase tag if necessary
        if (++sq->tail % sq->qs == 0)
        {
            sq->phase = !sq->phase;
        }

        return cmd;
    }

    return NULL;
}



/*
 * Enqueue command the i'th of n threads.
 *
 * This function does not check actual queue state, the caller should store
 * the pointer it last received and pass to the next call in order to simplify
 * position calculation.
 *
 * It is therefore important that all completions are consumed before clling
 * this function. 
 *
 * The reason for this is to avoid unecessary thread-synchronisation/barriers.
 *
 * Note: n must be less than the queue size
 * 
 * Note: The pointer should be stored and used as the last parameter for the
 *       succeeding call.
 */
#ifdef __CUDACC__
__device__ static inline
nvm_cmd_t* nvm_sq_enqueue_n(nvm_queue_t* sq, nvm_cmd_t* last, uint16_t n, uint16_t i)
{
    unsigned char* start = (unsigned char*) sq->vaddr;
    unsigned char* end = start + (sq->qs * sq->es);
    nvm_cmd_t* cmd = NULL;

    if (last == NULL)
    {
        cmd = (nvm_cmd_t*) (start + sq->es * i);
    }
    else
    {
        cmd = (nvm_cmd_t*) (((unsigned char*) last) + n * sq->es);

        if (((nvm_cmd_t*) end) <= cmd)
        {
            cmd = (nvm_cmd_t*) (start + (((unsigned char*) cmd) - end));
        }
    }

    // The 0'th thread should update the state
    if (i == 0)
    {
        sq->tail += n;
    }

    // Wait state updating here
    __syncthreads();

    return cmd;
}
#endif



/* 
 * Poll completion queue.
 *
 * Check the head of a completion queue for a new entry. The caller must
 * call dequeue manually.
 *
 * Returns a pointer to an unprocessed completion entry, or NULL if the queue
 * is empty.
 */
__host__ __device__ static inline
nvm_cpl_t* nvm_cq_poll(const nvm_queue_t* cq)
{
    nvm_cpl_t* cpl = (nvm_cpl_t*) (((unsigned char*) cq->vaddr) + cq->es * (cq->head % cq->qs));

#ifndef __CUDA_ARCH__
    if (cq->local) 
    {
        nvm_cache_invalidate((void*) cpl, sizeof(nvm_cpl_t));
    }
#endif

    // Check if new completion is ready by checking the phase tag
    if (!_RB(*NVM_CPL_STATUS(cpl), 0, 0) != !cq->phase)
    {
        return NULL;
    }

    return cpl;
}



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
__host__ __device__ static inline
nvm_cpl_t* nvm_cq_dequeue(nvm_queue_t* cq)
{
    nvm_cpl_t* cpl = nvm_cq_poll(cq);

    if (cpl != NULL)
    {
        // Increase head pointer and invert phase tag
        if (++cq->head % cq->qs == 0)
        {
            cq->phase = !cq->phase;
        }
    }

    return cpl;
}



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
#ifdef __cplusplus
extern "C" {
#endif
__host__
nvm_cpl_t* nvm_cq_dequeue_block(nvm_queue_t* cq, uint64_t timeout);
#ifdef __cplusplus
}
#endif



/* 
 * Update SQ tail pointer.
 *
 * Submit all enqueued commands by ringing the doorbell.
 * The caller must make sure that all commands are prepared before calling
 * this.
 */
__host__ __device__ static inline
void nvm_sq_submit(nvm_queue_t* sq)
{
    const uint16_t t = sq->tail % sq->qs;

    if (sq->last != t && sq->db != NULL)
    {
#ifndef __CUDA_ARCH__
        if (sq->local)
        {
            nvm_cache_flush((void*) sq->vaddr, sq->es * sq->qs);
        }
        else 
        {
            nvm_wcb_flush(); 
        }
#endif

        *((volatile uint32_t*) sq->db) = t;
        sq->last = t;
    }
}



/* 
 * Update SQ head pointer.
 */
__host__ __device__ static inline
void nvm_sq_update(nvm_queue_t* sq)
{
    // Update head pointer of submission queue
    if (sq->db != NULL)
    {
        ++sq->head;
    }
}



/* 
 * Update controller's CQ head pointer.
 *
 * Indicate that all completions are processed by ringing the doorbell.
 * All completion pointers acquired before this must be discarded after
 * calling this.
 */
__host__ __device__ static inline
void nvm_cq_update(nvm_queue_t* cq)
{
    if (cq->last != cq->head % cq->qs && cq->db != NULL)
    {
        cq->last = cq->head % cq->qs;
        *((volatile uint32_t*) cq->db) = cq->last;
        cq->tail = cq->head;
    }
}


#ifndef __CUDACC__
#undef __device__
#undef __host__
#endif

#endif /* __NVM_QUEUE_H__ */
