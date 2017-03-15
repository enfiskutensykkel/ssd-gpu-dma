#include "nvme.h"
#include "nvme_init.h"
#include "nvme_core.h"
#include "nvme_queue.h"
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include "util.h"


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

#define CSTS$RDY(p)     _RB(*CSTS(p), 0,  0)    // Ready indicator


/* Write bit fields */
#define CC$IOCQES(v)    _WB(v, 23, 20)          // IO Completion Queue Entry Size
#define CC$IOSQES(v)    _WB(v, 19, 16)          // IO Submission Queue Entry Size
#define CC$MPS(v)       _WB(v, 10,  7)          // Memory Page Size
#define CC$CSS(v)       _WB(0,  3,  1)          // IO Command Set Selected (0=NVM Command Set)
#define CC$EN(v)        _WB(v,  0,  0)          // Enable

#define AQA$AQS(v)      _WB(v, 27, 16)          // Admin Completion Queue Size
#define AQA$AQC(v)      _WB(v, 11,  0)          // Admin Submission Queue Size


/* List of admin command opcodes */
enum 
{
    IDENTIFY_CONTROLLER     = (0x00 << 7) | (0x01 << 2) | 0x02,
    GET_FEATURES            = (0x00 << 7) | (0x02 << 2) | 0x02,
    SET_FEATURES            = (0x00 << 7) | (0x02 << 2) | 0x01
};


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


static void configure_entry_sizes(volatile void* register_ptr, nvm_controller_t controller)
{
    volatile uint32_t* cc = CC(register_ptr);
    *cc |= CC$IOCQES(controller->cq_entry_size) | CC$IOSQES(controller->sq_entry_size);
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


static int identify_controller(nvm_controller_t controller, volatile void* register_ptr)
{
    struct command* identify_cmd = sq_enqueue(controller->queue_handles[0]);
    if (identify_cmd  == NULL)
    {
        return -ENOSPC;
    }

    struct command* get_features_cmd = sq_enqueue(controller->queue_handles[0]);
    if (get_features_cmd == NULL)
    {
        return -ENOSPC;
    }

    uint16_t feature_id = *COMMAND_ID(get_features_cmd);

    identify_cmd->dword[0] |= (0 << 14) | (0 << 8) | IDENTIFY_CONTROLLER;
    identify_cmd->dword[1] = 0xffffffff; // FIXME: should this be 0?
    identify_cmd->dword[10] = (0 << 16) | 1;

    uint64_t buff_addr = controller->data.phys_addr;
    identify_cmd->dword[6] = (uint32_t) buff_addr;
    identify_cmd->dword[7] = (uint32_t) (buff_addr >> 32);
    identify_cmd->dword[8] = 0;
    identify_cmd->dword[9] = 0;

    get_features_cmd->dword[0] |= (0 << 14) | (0 << 8) | GET_FEATURES;
    get_features_cmd->dword[1] = 0; // We think this is supposed to be 0

    get_features_cmd->dword[10] = (0x03 << 8) | 0x07;
    get_features_cmd->dword[11] = 0;

    sq_submit(controller->queue_handles[0]);

    // Wait for completions
    size_t completions_left = 2;
    while (completions_left > 0)
    {
        struct completion* cpl;

        // Poll until there are any completions
        while ((cpl = cq_poll(controller->queue_handles[1])) == NULL);

        // Consume completions
        while ((cpl = cq_dequeue(controller->queue_handles[1], controller)) != NULL)
        {
            // Update the maximum number of supported queues
            if (*COMPLETION_ID(cpl) == feature_id)
            {
                uint16_t max_queues = _MIN(cpl->dword[0] >> 16, cpl->dword[0] & 0xffff);
                controller->max_queues = _MIN(max_queues, controller->max_queues);
            }

            --completions_left;
        }

        cq_submit(controller->queue_handles[1]);
    }

    unsigned char* bytes = controller->data.virt_addr;
    controller->max_data_size = bytes[77] * (1 << (12 + CAP$MPSMIN(register_ptr)));
    controller->sq_entry_size = _RB(bytes[512], 3, 0);
    controller->cq_entry_size = _RB(bytes[513], 3, 0);
    controller->max_out_cmds = *((uint16_t*) (bytes + 514));
    controller->n_ns = *((uint32_t*) (bytes + 516));
    return 0;
}


static int set_num_queues(nvm_controller_t controller)
{
    struct command* cmd = sq_enqueue(controller->queue_handles[0]);
    if (cmd == NULL)
    {
        return -ENOSPC;
    }

    cmd->dword[0] |= (0 << 14) | (0 << 8) | SET_FEATURES;
    cmd->dword[1] = 0;

    cmd->dword[10] = (0 << 31) | 0x07;
    cmd->dword[11] = (controller->max_queues << 16) | controller->max_queues;

    sq_submit(controller->queue_handles[0]);

    struct completion* cpl;
    while ((cpl = cq_dequeue(controller->queue_handles[1], controller)) == NULL);

    cq_submit(controller->queue_handles[1]);

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

    // Allocate buffer for controller data
    int err = get_page(&controller->data, fd, -1);
    if (err != 0)
    {
        fprintf(stderr, "Failed to allocate controller data page: %s\n", strerror(err));
        nvm_free(controller, fd);
        return err;
    }

    // Set controller properties
    controller->page_size = page_size;
    controller->dstrd = CAP$DSTRD(register_ptr);
    controller->enabled = 0;
    controller->timeout = CAP$TO(register_ptr) * 500UL;
    controller->max_data_size = 0;
    controller->max_entries = CAP$MQES(register_ptr) + 1;   // CAP.MQES is a 0's based value
    controller->cq_entry_size = sizeof(struct completion);
    controller->sq_entry_size = sizeof(struct command);
    controller->max_queues = (uint16_t) (db_size / (4 << controller->dstrd));
    controller->n_queues = 0;
    controller->queue_handles = NULL;
    controller->n_ns = 0;

    // Allocate queue handle table
    controller->queue_handles = malloc(sizeof(struct nvme_queue*) * controller->max_queues);
    if (controller->queue_handles == NULL)
    {
        fprintf(stderr, "Failed to allocate queue handle table: %s\n", strerror(errno));
        nvm_free(controller, fd);
        return errno;
    }

    // Reset queues
    for (uint16_t i = 0; i < controller->max_queues; ++i)
    {
        controller->queue_handles[i] = NULL;
    }

    // Create admin submission/completion queue pair
    err = create_queue_pair(controller, register_ptr);
    if (err != 0)
    {
        fprintf(stderr, "Failed to allocate admin queue handles\n");
        nvm_free(controller, fd);
        return err;
    }
    
    err = get_page(&controller->queue_handles[0]->page, fd, -1);
    if (err != 0)
    {
        fprintf(stderr, "Failed to allocate and pin admin queue memory: %s\n", strerror(err));
        nvm_free(controller, fd);
        return err;
    }

    err = get_page(&controller->queue_handles[1]->page, fd, -1);
    if (err != 0)
    {
        fprintf(stderr, "Failed to allocate and pin admin queue memory: %s\n", strerror(err));
        nvm_free(controller, fd);
        return err;
    }

    memset(controller->queue_handles[0]->page.virt_addr, 0, controller->queue_handles[0]->page.page_size);
    memset(controller->queue_handles[1]->page.virt_addr, 0, controller->queue_handles[1]->page.page_size);

    // Reset controller
    reset_controller(register_ptr, controller->timeout);

    // Set admin CQ and SQ
    configure_admin_queues(register_ptr, controller);

    // Bring controller back up
    enable_controller(register_ptr, host_page_size, controller->timeout);

    // Submit identify controller command
    err = identify_controller(controller, register_ptr);
    if (err != 0)
    {
        fprintf(stderr, "Failed to submit command: %s\n", strerror(err));
        nvm_free(controller, fd);
        return err;
    }

    // Set CQES and SQES in CC
    configure_entry_sizes(register_ptr, controller);

    // Set features
    err = set_num_queues(controller);
    if (err != 0)
    {
        fprintf(stderr, "Failed to submit command: %s\n", strerror(err));
        nvm_free(controller, fd);
        return err;
    }

    *handle = controller;
    return 0;
}


void nvm_free(nvm_controller_t handle, int ioctl_fd)
{
    put_page(&handle->data, ioctl_fd);

    if (handle->queue_handles != NULL)
    {
        for (uint16_t i = 2; i < handle->n_queues; ++i)
        {
            // TODO: submit stop processing command
            //delete_queue(handle, i);
        }

        // TODO: clean this up
        for (uint16_t i = 0; i < 2; ++i)
        {
            if (handle->queue_handles[i])
            {
                put_page(&handle->queue_handles[i]->page, ioctl_fd);
            }

            free(handle->queue_handles[i]);
        }

        free(handle->queue_handles);
    }

    free(handle);
}
