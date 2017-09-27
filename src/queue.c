#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include <errno.h>
#include "regs.h"
#include "util.h"


#define PHASE(p)    _RB(*CPL_STATUS(p),  0,  0) // Offset to phase tag bit


void nvm_queue_clear(nvm_queue_t* queue, const struct nvm_controller* ctrl, int cq, uint16_t no, void* vaddr, uint64_t ioaddr)
{
    queue->no = no;
    queue->max_entries = 0;
    queue->entry_size = cq ? sizeof(nvm_cpl_t) : sizeof(nvm_cmd_t);
    queue->head = 0;
    queue->tail = 0;
    queue->phase = 1;
    queue->last = 0;
    queue->vaddr = vaddr;
    queue->ioaddr = ioaddr;
    queue->db = cq ? CQ_DBL(ctrl->mm_ptr, queue->no, ctrl->dstrd) : SQ_DBL(ctrl->mm_ptr, queue->no, ctrl->dstrd);
    queue->max_entries = _MIN(ctrl->max_entries, ctrl->page_size / queue->entry_size);
}


nvm_cmd_t* sq_enqueue(nvm_queue_t* sq)
{
    // Check the capacity
    if ((uint16_t) ((sq->tail - sq->head) % sq->max_entries) == sq->max_entries - 1)
    {
        return NULL;
    }

    // Get slot at end of queue
    nvm_cmd_t* ptr = (nvm_cmd_t*) (((unsigned char*) sq->vaddr) + sq->entry_size * sq->tail);

    // Increase tail pointer and wrap around if necessary
    if (++sq->tail >= sq->max_entries)
    {
        sq->phase = !sq->phase;
        sq->tail = 0;
    }

    // Set command identifier to equal tail pointer
    // The caller may override this by manually setting the CID field in DWORD0
    *CMD_CID(ptr) = sq->tail + (!sq->phase) * sq->max_entries;

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


nvm_cpl_t* cq_dequeue_block(nvm_queue_t* cq, uint64_t timeout)
{
    uint64_t nsecs = timeout * 1000000UL;
    nvm_cpl_t* cpl = cq_dequeue(cq);

    while (cpl == NULL && nsecs > 0)
    {
        nsecs = _nvm_delay_remain(nsecs);
        cpl = cq_dequeue(cq);
    }

    return cpl;
}


void sq_submit(nvm_queue_t* sq)
{
    if (sq->last != sq->tail)
    {
        *((volatile uint32_t*) sq->db) = sq->tail;
        sq->last = sq->tail;
    }
}


void cq_update(nvm_queue_t* cq)
{
    if (cq->last != cq->head)
    {
        *((volatile uint32_t*) cq->db) = cq->head;
        cq->last = cq->head;
    }
}


void sq_update_unchecked(nvm_queue_t* sq)
{
    // Update head pointer of submission queue
    if (++sq->head >= sq->max_entries)
    {
        sq->head = 0;
    }
}


int sq_update(nvm_queue_t* sq, const nvm_cpl_t* cpl)
{
    if (cpl == NULL)
    {
        return EAGAIN;
    }

    if (sq->no != *CPL_SQID(cpl))
    {
        return EBADF;
    }

    sq_update_unchecked(sq);
    return 0;
}

