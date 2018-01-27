#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include "regs.h"
#include "util.h"



void nvm_queue_clear(nvm_queue_t* queue, const nvm_ctrl_t* ctrl, bool cq, uint16_t no, void* vaddr, uint64_t ioaddr)
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



nvm_cpl_t* nvm_cq_dequeue_block(nvm_queue_t* cq, uint64_t timeout)
{
    uint64_t nsecs = timeout * 1000000UL;
    nvm_cpl_t* cpl = nvm_cq_dequeue(cq);

    while (cpl == NULL && nsecs > 0)
    {
        nsecs = _nvm_delay_remain(nsecs);
        cpl = nvm_cq_dequeue(cq);
    }

    return cpl;
}

