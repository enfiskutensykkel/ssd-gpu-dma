#include "nvme_core.h"
#include "util.h"
#include "nvme.h"


// Bitfields in the completion structure
#define CQ_SQHD(p)          _REG(p,  8, 16)
#define CQ_SQID(p)          _REG(p, 10, 16)
#define CQ_STATUS(p)        _REG(p, 14, 16)
#define CQ_CMDID(p)         _REG(p, 12, 16)


// Offset to phase tag bit
#define PHASE(p)            _RB(*CQ_STATUS(p),  0,  0)


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
    ptr->dword[0] = sq->tail << 16;

    return ptr;
}


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


struct completion* cq_dequeue(nvm_queue_t cq, nvm_controller_t controller)
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

        // Update head pointer of submission queue
        controller->queue_handles[*CQ_SQID(ptr)]->head = *CQ_SQHD(ptr);
    }

    return ptr;
}


void sq_submit(nvm_queue_t sq)
{
    *sq->db = sq->tail;
}


void cq_submit(nvm_queue_t cq)
{
    *cq->db = cq->head;
}

