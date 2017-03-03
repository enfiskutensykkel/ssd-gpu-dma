#include "nvme.h"
#include "nvme_init.h"
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>


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
    controller->max_queues = db_size / (4 << controller->dstrd);
    controller->n_queues = 0;
    controller->queue_handles = NULL;

    // Allocate queue handle table
    controller->queue_handles = malloc(sizeof(struct nvm_queue) * controller->max_queues);
    if (controller->queue_handles == NULL)
    {
        fprintf(stderr, "Failed to allocate queue handle table: %s\n", strerror(errno));
        return errno;
    }

    for (size_t i = 0; i < controller->max_queues; ++i)
    {
        controller->queue_handles[i] = NULL;
    }

    // TODO: Create admin SQ and CQ and page lock them

    // Reset controller
    reset_controller(register_ptr, controller->timeout);

    // Set admin CQ and SQ
    //configure_admin_queues();

    // Bring controller back up
    enable_controller(register_ptr, host_page_size, controller->timeout);

    *handle = controller;
    fprintf(stderr, "NVMe controller initiated\n");
    return 0;
}


void nvm_free(nvm_controller_t handle)
{
    free(handle->queue_handles);
    free(handle);
}
