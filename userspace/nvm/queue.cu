#include <cuda.h>
#include "types.h"
#include "queue.h"
#include "command.h"
#include "util.h"
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <errno.h>


#define PHASE(p)    _RB(*CPL_STATUS(p),  0,  0) // Offset to phase tag bit


extern "C" __host__ __device__
struct command* sq_enqueue(nvm_queue_t* sq)
{
    // Check the capacity
    if ((sq->tail - sq->head) % sq->max_entries == sq->max_entries - 1)
    {
        return NULL;
    }

    // Get slot at end of queue
    struct command* ptr = 
        (struct command*) (((unsigned char*) sq->virt_addr) + sq->entry_size * sq->tail);

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


extern "C" __host__ __device__
struct completion* cq_poll(const nvm_queue_t* cq)
{
    struct completion* ptr = 
        (struct completion*) (((unsigned char*) cq->virt_addr) + cq->entry_size * cq->head);

    // Check if new completion is ready by checking the phase tag
    if (!!PHASE(ptr) != cq->phase)
    {
        return NULL;
    }

    return ptr;
}


extern "C" __host__ __device__
struct completion* cq_dequeue(nvm_queue_t* cq)
{
    struct completion* ptr = cq_poll(cq);

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
__host__
static inline void delay(uint64_t& remaining_nsecs)
{
    if (remaining_nsecs == 0)
    {
        return;
    }

    timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = _MIN(1000000UL, remaining_nsecs);

    clock_nanosleep(CLOCK_REALTIME, 0, &ts, NULL);

    remaining_nsecs -= _MIN(1000000UL, remaining_nsecs);
}


extern "C" __host__ 
struct completion* cq_dequeue_block(nvm_queue_t* cq, uint64_t timeout)
{
    uint64_t nsecs = timeout * 1000000UL;
    struct completion* cpl = cq_dequeue(cq);

    while (cpl == NULL && nsecs > 0)
    {
        delay(nsecs);
        cpl = cq_dequeue(cq);
    }

    return cpl;
}


extern "C" __host__ __device__
void sq_submit(const nvm_queue_t* sq)
{
    *((volatile uint32_t*) sq->db) = sq->tail;
}


extern "C" __host__ __device__
void cq_update(const nvm_queue_t* cq)
{
    *((volatile uint32_t*) cq->db) = cq->head;
}


extern "C" __host__ __device__
int sq_update(nvm_queue_t* sq, const struct completion* cpl)
{
    if (cpl == NULL)
    {
        return EAGAIN;
    }

    if (sq->no == *CPL_SQID(cpl))
    {
        // Update head pointer of submission queue
        sq->head = *CPL_SQHD(cpl);
        return 0;
    }

    return EBADF;
}

