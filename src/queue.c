#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include "regs.h"
#include "util.h"



int nvm_queue_clear(nvm_queue_t* queue, const nvm_ctrl_t* ctrl, bool cq, uint16_t no, uint32_t qs, 
        bool local, volatile void* vaddr, uint64_t ioaddr)
{
    if (qs < 2 || qs > 0x10000 || qs > ctrl->max_qs)
    {
        return EINVAL;
    }

    queue->no = no;
    queue->qs = qs;
    queue->es = cq ? sizeof(nvm_cpl_t) : sizeof(nvm_cmd_t);
    queue->head = 0;
    queue->tail = 0;
    queue->last = 0;
    queue->phase = 1;
    queue->local = !!local;
    queue->db = cq ? CQ_DBL(ctrl->mm_ptr, queue->no, ctrl->dstrd) : SQ_DBL(ctrl->mm_ptr, queue->no, ctrl->dstrd);
    queue->vaddr = vaddr;
    queue->ioaddr = ioaddr;
    
    return 0;
}



void nvm_queue_reset(nvm_queue_t* queue)
{
    queue->head = 0;
    queue->tail = 0;
    queue->last = 0;
    queue->phase = 1;
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

