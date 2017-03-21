#include <cuda.h>
#include "types.h"
#include "queue.h"
#include "command.h"
#include "util.h"
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <errno.h>


// Offset to phase tag bit
#define PHASE(p)            _RB(*CPL_STATUS(p),  0,  0)


extern "C" __host__ __device__
int cmd_data_ptr(struct command* cmd, memory_t* prp_list, memory_t* prps, size_t n_prps)
{
    cmd->dword[0] &= ~( (0x03 << 14) | (0x03 << 8) );

    if (n_prps > prps->n_addrs)
    {
        return ENOSPC;
    }

    cmd->dword[6] = (uint32_t) prps->bus_addr[0];
    cmd->dword[7] = (uint32_t) (prps->bus_addr[0] >> 32);

    if (n_prps <= 1)
    {
        cmd->dword[8] = 0;
        cmd->dword[9] = 0;
    }
    else if (n_prps == 2)
    {
        cmd->dword[8] = (uint32_t) prps->bus_addr[1];
        cmd->dword[9] = (uint32_t) (prps->bus_addr[1] >> 32);
    }
    else
    {
        // TODO Implement PRP list handling
    }

    return 0;
}


extern "C" __host__ __device__
void cmd_header(struct command* cmd, uint8_t opcode, uint32_t ns_id)
{
    cmd->dword[0] &= 0xffff0000;
    cmd->dword[0] |= (0x00 << 14) | (0x00 << 8) | (opcode & 0x7f);
    cmd->dword[1] = ns_id;
}


extern "C" __host__ __device__
struct command* sq_enqueue(nvm_queue_t sq)
{
    // Check the capacity
    if (sq->tail - sq->head >= sq->max_entries)
    {
        return NULL;
    }

    // Get slot at end of queue
    struct command* ptr = (struct command*) (((unsigned char*) sq->page.virt_addr) + sq->entry_size * sq->tail);

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
struct completion* cq_poll(nvm_queue_t cq)
{
    struct completion* ptr = 
        (struct completion*) (((unsigned char*) cq->page.virt_addr) + cq->entry_size * cq->head);

    // Check if new completion is ready by checking the phase tag
    if (PHASE(ptr) != cq->phase)
    {
        return NULL;
    }

    return ptr;
}


extern "C" __host__ __device__
struct completion* cq_dequeue(nvm_queue_t cq, nvm_controller_t ctrl)
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

        if (ctrl != NULL)
        {
            // Update head pointer of submission queue
            ctrl->queues[*CPL_SQID(ptr)]->head = *CPL_SQHD(ptr);
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
struct completion* cq_dequeue_block(nvm_queue_t cq, nvm_controller_t ctrl)
{
    uint64_t nsecs = ctrl->timeout * 1000000UL;
    struct completion* cpl = cq_dequeue(cq, ctrl);

    while (cpl == NULL && nsecs > 0)
    {
        delay(nsecs);
        cpl = cq_dequeue(cq, ctrl);
    }

    return cpl;
}


extern "C" __host__ __device__
void sq_submit(nvm_queue_t sq)
{
    *((volatile uint32_t*) sq->db) = sq->tail;
}


extern "C" __host__ __device__
void cq_update(nvm_queue_t cq)
{
    *((volatile uint32_t*) cq->db) = cq->head;
}

