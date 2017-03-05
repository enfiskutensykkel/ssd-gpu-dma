#include "nvme.h"
#include "nvme_init.h"
#include "nvme_core.h"
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <time.h>


#define _MIN(a, b) ( (a) <= (b) ? (a) : (b) )

/* Convenience function for creating a bit mask */
static inline uint64_t bitmask(int hi, int lo)
{
    uint64_t mask = 0;

    for (int i = lo; i <= hi; ++i)
    {
        mask |= 1UL << i;
    }

    return mask;
}


/* Extract specific bits */
#define _RB(v, hi, lo)   \
    ( ( (v) & bitmask((hi), (lo)) ) >> (lo) )


/* Set specifics bits */
#define _WB(v, hi, lo)   \
    ( ( (v) << (lo) ) & bitmask((hi), (lo)) )


/* Offset to a register */
#define _REG(p, offs, bits) \
    ((volatile uint##bits##_t *) (((volatile unsigned char*) (p)) + (offs)))


/* Controller registers */
#define CAP(p)          _REG(p, 0x0000, 64)     // Controller Capabilities
#define CC(p)           _REG(p, 0x0014, 32)     // Controller Configuration
#define CSTS(p)         _REG(p, 0x001c, 32)     // Controller Status
#define AQA(p)          _REG(p, 0x0024, 32)     // Admin Queue Attributes
#define ASQ(p)          _REG(p, 0x0028, 64)     // Admin Submission Queue Base Address
#define ACQ(p)          _REG(p, 0x0030, 64)     // Admin Completion Queue Base Address


/* Read bit fields */
#define CAP$MPSMAX(p)   _RB(*CAP(p), 55, 52)    // Memory Page Size Maximum
#define CAP$MPSMIN(p)   _RB(*CAP(p), 51, 48)    // Memory Page Size Minimum
#define CAP$DSTRD(p)    _RB(*CAP(p), 35, 32)    // Doorbell Stride
#define CAP$TO(p)       _RB(*CAP(p), 31, 24)    // Timeout
#define CAP$CQR(p)      _RB(*CAP(p), 16, 16)    // Contiguous Queues Required
#define CAP$MQES(p)     _RB(*CAP(p), 15,  0)    // Maximum Queue Entries Supported

#define CSTS$RDY(p)     _RB(*CSTS(p), 0,  0)    // Ready


/* Write bit fields */
#define CC$IOCQES(v)    _WB(v, 23, 20)          // IO Completion Queue Entry Size
#define CC$IOSQES(v)    _WB(v, 19, 16)          // IO Submission Queue Entry Size
#define CC$MPS(v)       _WB(v, 10,  7)          // Memory Page Size
#define CC$CSS(v)       _WB(0,  3,  1)          // IO Command Set Selected (0=NVM Command Set)
#define CC$EN(v)        _WB(v,  0,  0)          // Enable

#define AQA$AQS(v)      _WB(v, 27, 16)          // Admin Completion Queue Size
#define AQA$AQC(v)      _WB(v, 11,  0)          // Admin Submission Queue Size


/* Encode page size to a the format required by the controller */
static uint8_t encode_page_size(size_t page_size)
{
    page_size >>= 12;
    size_t count = 0;

    while (page_size > 0)
    {
        ++count;
        page_size >>= 1;
    }

    return count - 1;
}


/* Delay execution by 1 millisecond */
static uint64_t delay(uint64_t timeout, uint64_t remaining, uint64_t reset)
{
    struct timespec ts;
    ts.tv_sec = timeout / 1000;
    ts.tv_nsec = 1000000UL * (timeout % 1000);

    nanosleep(&ts, NULL);

    if (remaining == 0)
    {
        return reset; 
    }
    
    if (remaining < timeout)
    {
        return 0;
    }

    return remaining - timeout;
}


static int reset_controller(volatile void* register_ptr, uint64_t timeout)
{
    volatile uint32_t* cc = CC(register_ptr);

    // Set CC.EN to 0
    *cc = *cc & ~1;

    // Wait for CSTS.RDY to transition from 1 to 0
    uint64_t remaining = delay(1, 0, timeout);

    while (CSTS$RDY(register_ptr) != 0)
    {
        if (remaining == 0)
        {
            fprintf(stderr, "Timeout exceeded while waiting for CSTS.RDY 1 -> 0\n");
            return ETIME;
        }

        remaining = delay(1, remaining, timeout);
    }

    return 0;
}


static int enable_controller(volatile void* register_ptr, uint8_t encoded_page_size, uint64_t timeout)
{
    volatile uint32_t* cc = CC(register_ptr);

    // Set CC.MPS = <pagesize> and CC.EN to 1 
    *cc = CC$IOCQES(0) | CC$IOSQES(0) | CC$MPS(encoded_page_size) | CC$CSS(0) | CC$EN(1);

    // Wait for CSTS.RDY to transition from 0 to 1
    uint64_t remaining = delay(1, 0, timeout);

    while (CSTS$RDY(register_ptr) != 1)
    {
        if (remaining == 0)
        {
            fprintf(stderr, "Timeout exceeded while waiting for CSTS.RDY 0 -> 1\n");
            return ETIME;
        }

        remaining = delay(1, remaining, timeout);
    }

    return 0;
}


static nvm_queue_t alloc_admin_queue(int ioctl_fd, unsigned int no, size_t entry_size)
{
    // Allocate queue handle
    nvm_queue_t queue = malloc(sizeof(struct nvm_queue));
    if (queue == NULL)
    {
        fprintf(stderr, "Failed to allocate queue handle: %s\n", strerror(errno));
        return NULL;
    }

    // Create page of memory
    int err = get_page(&queue->page, ioctl_fd, -1);
    if (err != 0)
    {
        fprintf(stderr, "Failed to allocate and pin queue memory: %s\n", strerror(err));
        free(queue);
        return NULL;
    }

    queue->no = no;
    queue->max_entries = _MIN(16, queue->page.page_size / entry_size);
    queue->entry_size = entry_size;
    queue->head = 0;
    queue->tail = 0;
    queue->phase = 0;
    queue->db = NULL;

    memset(queue->page.virt_addr, 0, queue->page.page_size);

    return queue;
}


/* Set admin queue registers */
static void configure_admin_queues(volatile void* register_ptr, nvm_controller_t controller)
{
    nvm_queue_t sq = controller->queue_handles[0];
    nvm_queue_t cq = controller->queue_handles[1];

    volatile uint32_t* aqa = AQA(register_ptr);
    *aqa = AQA$AQS(sq->max_entries - 1) | AQA$AQC(cq->max_entries - 1);

    volatile uint64_t* asq = ASQ(register_ptr);
    *asq = sq->page.phys_addr;

    volatile uint64_t* acq = ACQ(register_ptr);
    *acq = cq->page.phys_addr;
}


/* Send an identify controller NVM command */
static void identify_controller(nvm_controller_t controller, int ioctl_fd)
{
    page_t identify_data;
    int err = get_page(&identify_data, ioctl_fd, -1);
    if (err != 0)
    {
        fprintf(stderr, "Failed to allocate buffer for identify command\n");
        return;
    }

    struct {
        uint32_t dword[16];
    } command;

    volatile struct {
        uint32_t dword[4];
    } *completion;
    memset(&command, 0, sizeof(command));

    uint32_t opcode = (0 << 7) | (0x1 << 2) | (0x2);
    command.dword[0] = (0xbeef << 16) | (0 << 14) | (0 << 8) | opcode;
    command.dword[1] = 0xffffffff;

    command.dword[4] = 0x0;
    command.dword[5] = 0x0;

    uint64_t buf_addr = identify_data.phys_addr;
    command.dword[7] = (uint32_t) (buf_addr >> 32);
    command.dword[6] = (uint32_t) buf_addr;
    command.dword[8] = 0;
    command.dword[9] = 0;

    uint8_t cns = 1;
    command.dword[10] = (0 << 16) | cns;

    nvm_queue_t sq = controller->queue_handles[0];
    memcpy(sq->page.virt_addr, &command, sizeof(command));
    sq->tail += 1;
    *(sq->db) = sq->tail;
    __sync_synchronize();

    nvm_queue_t cq = controller->queue_handles[1];
    completion = cq->page.virt_addr;

    for (size_t i = 0; i < 3; ++i)
    {
        delay(1500, 0, 0);
        printf("0: %x\n", completion->dword[0]);
        printf("1: %x\n", completion->dword[1]);
        printf("2: %x\n", completion->dword[2]);
        printf("3: %x\n", completion->dword[3]);
        printf("\n");
    }
}


int nvm_init(nvm_controller_t* handle, int fd, volatile void* register_ptr, size_t db_size)
{
    *handle = NULL;
    nvm_controller_t controller;

    // Read out controller capabilities of interest
    long page_size = sysconf(_SC_PAGESIZE);

    if (page_size == -1)
    {
        fprintf(stderr, "Failed to retrieve host page size: %s\n", strerror(errno));
        return errno;
    }

    uint8_t host_page_size = encode_page_size(page_size);
    uint8_t max_page_size = CAP$MPSMAX(register_ptr);
    uint8_t min_page_size = CAP$MPSMIN(register_ptr);

    if (!(min_page_size <= host_page_size && host_page_size <= max_page_size))
    {
        fprintf(stderr, "Host is configured with an unsupported page size\n");
        return ENOSPC;
    }

    // Allocate controller handle
    controller = malloc(sizeof(struct nvm_controller));
    if (controller == NULL)
    {
        fprintf(stderr, "Failed to allocate controller structure: %s\n", strerror(errno));
        return errno;
    }

    // Set controller properties
    controller->page_size = page_size;
    controller->dstrd = CAP$DSTRD(register_ptr);
    controller->enabled = 0;
    controller->timeout = CAP$TO(register_ptr) * 500UL;
    controller->max_entries = CAP$MQES(register_ptr) + 1;   // CAP.MQES is a 0's based value
    controller->cq_entry_size = 0;
    controller->sq_entry_size = 0;
    controller->max_queues = (int16_t) (db_size / (4 << controller->dstrd));
    controller->n_queues = 0;
    controller->queue_handles = NULL;

    // Allocate queue handle table
    controller->queue_handles = malloc(sizeof(struct nvm_queue) * controller->max_queues);
    if (controller->queue_handles == NULL)
    {
        fprintf(stderr, "Failed to allocate queue handle table: %s\n", strerror(errno));
        nvm_free(controller, fd);
        return errno;
    }

    // Reset queues
    for (int16_t i = 0; i < controller->max_queues; ++i)
    {
        controller->queue_handles[i] = NULL;
    }

    // Create admin submission/completion queue pair
    controller->n_queues = 2;
    controller->queue_handles[0] = alloc_admin_queue(fd, 0, 64);
    controller->queue_handles[1] = alloc_admin_queue(fd, 0, 16);

    if (controller->queue_handles[0] == NULL || controller->queue_handles[1] == NULL)
    {
        fprintf(stderr, "Failed to allocate admin queues\n");
        nvm_free(controller, fd);
        return errno;
    }

    controller->queue_handles[0]->db = SQ_DBL(register_ptr, 0, controller->dstrd);
    controller->queue_handles[1]->db = CQ_DBL(register_ptr, 1, controller->dstrd);

    // Reset controller
    reset_controller(register_ptr, controller->timeout);

    // Set admin CQ and SQ
    configure_admin_queues(register_ptr, controller);

    // Bring controller back up
    enable_controller(register_ptr, host_page_size, controller->timeout);

    // Submit identify controller command
    identify_controller(controller, fd);

    *handle = controller;
    fprintf(stderr, "NVMe controller initiated\n");
    return 0;
}


void nvm_free(nvm_controller_t handle, int ioctl_fd)
{
    // TODO: submit stop processing command

    if (handle->queue_handles != NULL)
    {
        int16_t i;
        for (i = 0; i < 2; ++i)
        {
            if (handle->queue_handles[i])
            {
                put_page(&handle->queue_handles[i]->page, ioctl_fd);
            }

            free(handle->queue_handles[i]);
        }

        for (; i < handle->n_queues; ++i)
        {
            //delete_queue(handle, i);
        }

        free(handle->queue_handles);
    }

    free(handle);
}
