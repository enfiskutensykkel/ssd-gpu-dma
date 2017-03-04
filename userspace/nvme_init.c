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
static uint64_t delay(uint64_t remaining, uint64_t timeout)
{
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 1000000UL;

    nanosleep(&ts, NULL);

    if (remaining <= 0)
    {
        return timeout; 
    }
    
    return remaining - 1;
}


static int reset_controller(volatile void* register_ptr, uint64_t timeout)
{
    volatile uint32_t* cc = CC(register_ptr);

    // Set CC.EN to 0
    *cc = *cc & ~1;

    // Wait for CSTS.RDY to transition from 1 to 0
    uint64_t remaining = delay(0, timeout);

    while (CSTS$RDY(register_ptr) != 0)
    {
        if (remaining == 0)
        {
            fprintf(stderr, "Timeout exceeded while waiting for CSTS.RDY 1 -> 0\n");
            return ETIME;
        }

        remaining = delay(remaining, timeout);
    }

    return 0;
}


static int enable_controller(volatile void* register_ptr, uint8_t encoded_page_size, uint64_t timeout)
{
    volatile uint32_t* cc = CC(register_ptr);

    // Set CC.MPS = <pagesize> and CC.EN to 1 
    *cc = CC$IOCQES(0) | CC$IOSQES(0) | CC$MPS(encoded_page_size) | CC$CSS(0) | CC$EN(1);

    // Wait for CSTS.RDY to transition from 0 to 1
    uint64_t remaining = delay(0, timeout);

    while (CSTS$RDY(register_ptr) != 1)
    {
        if (remaining == 0)
        {
            fprintf(stderr, "Timeout exceeded while waiting for CSTS.RDY 0 -> 1\n");
            return ETIME;
        }

        remaining = delay(remaining, timeout);
    }

    return 0;
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
        nvm_free(controller);
        return errno;
    }

    for (int16_t i = 2; i < controller->max_queues; ++i)
    {
        controller->queue_handles[i] = NULL;
    }

    // Create admin submission queue
    page_t admin_sq;
    int err = get_page(&admin_sq, fd, -1);
    if (err != 0)
    {
        fprintf(stderr, "Failed to allocate and pin admin SQ: %s\n", strerror(err));
        nvm_free(controller);
        return err;
    }

    // TODO: Create admin SQ and CQ and page lock them

    // Reset controller
    reset_controller(register_ptr, controller->timeout);

    // Set admin CQ and SQ
    //configure_admin_queues();

    // Bring controller back up
    enable_controller(register_ptr, host_page_size, controller->timeout);

    // Submit identify controller command

    *handle = controller;
    fprintf(stderr, "NVMe controller initiated\n");
    return 0;
}


void nvm_free(nvm_controller_t handle)
{
    if (handle->queue_handles != NULL)
    {
        for (int16_t i = 2; i < handle->n_queues; ++i)
        {
            //delete_queue(handle, i);
        }

        free(handle->queue_handles);
    }

    free(handle);
}
