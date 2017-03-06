#include "nvme_core.h"
#include "nvme.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include "util.h"


#define CQ_SQID(p)      _REG(p,  8, 16)
#define CQ_SQHD(p)      _REG(p, 10, 16)
#define CQ_STATUS(p)    _REG(p, 12, 16)
#define CQ_CMDID(p)     _REG(p, 14, 16)
#define DNR(p)          _RB(*CQ_STATUS(p), 31, 31)
#define PHASE(p)        _RB(*CQ_STATUS(p),  0,  0)


/* NVM command data type (64 bytes) */
struct command
{
    uint32_t dword[16];
};


/* NVM completion data type (16 bytes) */
struct completion
{
    uint32_t dword[4];
};


int produce_command(nvm_queue_t sq, struct command* cmd)
{
    if (sq->tail - sq->head >= sq->max_entries)
    {
        return 0;
    }

    size_t pos = (sq->tail - sq->head) % sq->max_entries;

    struct command* ptr = (struct command*) (((unsigned char*) sq->page.virt_addr) + sq->entry_size * pos);
    memcpy(ptr, cmd, sizeof(struct command));

    if (++sq->tail == sq->max_entries)
    {
        sq->tail = 0;
    }

    // Set command identifier to equal tail pointer
    ptr->dword[0] &= (uint32_t) ~(0xffff << 16);
    ptr->dword[0] |= sq->tail << 16;

    // Ring the doorbell
    *sq->db = sq->tail;
    return 1;
}


int consume_completion(nvm_queue_t sq, nvm_queue_t cq, struct completion* cpl)
{
    struct completion* ptr = (struct completion*) (((unsigned char*) cq->page.virt_addr) + cq->entry_size * cq->head);

    if ((*CQ_CMDID(ptr) - cq->head) % cq->max_entries > cq->tail)
    {
        memcpy(cpl, ptr, sizeof(struct completion));

        sq->head = *CQ_CMDID(ptr);
        
        if (++cq->tail == cq->max_entries)
        {
            cq->tail = 0;
        }

        if (++cq->head == cq->max_entries)
        {
            cq->head = 0;
        }

        // Update head pointer to indicate that completion is read
        *cq->db = cq->head;
        return 1;
    }

    return 0;
}


/* Send an identify controller NVM command */
int identify_controller(nvm_controller_t controller, page_t* data)
{
    struct command cmd;
    struct completion cpl;
    memset(&cmd, 0, sizeof(cmd));
    memset(&cpl, 0, sizeof(cpl));

    uint32_t opcode = (0 << 7) | (0x1 << 2) | (0x2);
    cmd.dword[0] = (0 << 14) | (0 << 8) | opcode;
    cmd.dword[1] = 0xffffffff;

    cmd.dword[4] = 0x0;
    cmd.dword[5] = 0x0;

    uint64_t bufaddr = data->phys_addr;
    cmd.dword[7] = (uint32_t) (bufaddr >> 32);
    cmd.dword[6] = (uint32_t) bufaddr;
    cmd.dword[8] = 0;
    cmd.dword[9] = 0;

    uint8_t cns = 1;
    cmd.dword[10] = (0 << 16) | cns;

    // Submit command
    if (produce_command(controller->queue_handles[0], &cmd) != 1)
    {
        fprintf(stderr, "Couldn't enqueue NVM command\n");
        return -EAGAIN;
    }

    // Poll for completion
    while (consume_completion(controller->queue_handles[0], controller->queue_handles[1], &cpl) == 0);

    return 0;
}

