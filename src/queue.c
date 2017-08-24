#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include <errno.h>


#define PHASE(p)    _RB(*CPL_STATUS(p),  0,  0) // Offset to phase tag bit


nvm_cmd_t* sq_enqueue(nvm_queue_t* sq)
{
    // Check the capacity
    if ((sq->tail - sq->head) % sq->max_entries == sq->max_entries - 1)
    {
        return NULL;
    }

    // Get slot at end of queue
    nvm_cmd_t* ptr = (nvm_cmd_t*) (((unsigned char*) sq->vaddr) + sq->entry_size * sq->tail);

    // Increase tail pointer and wrap around if necessary
    if (++sq->tail >= sq->max_entries)
    {
        sq->tail = 0;
    }

    // Set command identifier to equal tail pointer
    // The caller may override this by manually setting the CID field in DWORD0
    *CMD_CID(ptr) = sq->tail;

    return ptr;
}


nvm_cpl_t* cq_poll(const nvm_queue_t* cq)
{
    nvm_cpl_t* ptr = (nvm_cpl_t*) (((unsigned char*) cq->vaddr) + cq->entry_size * cq->head);

    // Check if new completion is ready by checking the phase tag
    if (!!PHASE(ptr) != cq->phase)
    {
        return NULL;
    }

    return ptr;
}


nvm_cpl_t* cq_dequeue(nvm_queue_t* cq)
{
    nvm_cpl_t* ptr = cq_poll(cq);

    if (ptr != NULL)
    {
        // Increase head pointer and wrap around if necessary
        if (++cq->head >= cq->max_entries)
        {
            cq->head = 0;
            cq->phase = !cq->phase;
        }
    }

    return ptr;
}


/* Delay execution by one millisecond */
static inline void delay(uint64_t* remaining_nsecs)
{
    if (*remaining_nsecs == 0)
    {
        return;
    }

    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = _MIN(1000000UL, *remaining_nsecs);

    clock_nanosleep(CLOCK_REALTIME, 0, &ts, NULL);

    *remaining_nsecs -= _MIN(1000000UL, *remaining_nsecs);
}


nvm_cpl_t* cq_dequeue_block(nvm_queue_t* cq, uint64_t timeout)
{
    uint64_t nsecs = timeout * 1000000UL;
    nvm_cpl_t* cpl = cq_dequeue(cq);

    while (cpl == NULL && nsecs > 0)
    {
        delay(&nsecs);
        cpl = cq_dequeue(cq);
    }

    return cpl;
}


void sq_submit(const nvm_queue_t* sq)
{
    *((volatile uint32_t*) sq->db) = sq->tail;
}


void cq_update(const nvm_queue_t* cq)
{
    *((volatile uint32_t*) cq->db) = cq->head; // FIXME: Check that it only moves forward
}


int sq_update(nvm_queue_t* sq, const nvm_cpl_t* cpl)
{
    if (cpl == NULL)
    {
        return EAGAIN;
    }

    if (sq->no == *CPL_SQID(cpl))
    {
        // Update head pointer of submission queue
        sq->head = *CPL_SQHD(cpl); // FIXME: Check that it only moves forward
        return 0;
    }

    return EBADF;
}

