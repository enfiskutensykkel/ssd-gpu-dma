#include "types.h"
#include "util.h"
#include "queue.h"
#include "command.h"
#include "ctrl.h"
#include "memory/types.h"
#include "memory/ram.h"
#include "memory/gpu.h"
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <stdio.h>


/* SQ doorbell register */
#define SQ_DBL(p, y, dstrd)    \
    ((volatile void*) (((volatile unsigned char*) (p)) + 0x1000 + ((2*(y)) * (4 << (dstrd)))) )


/* CQ doorbell register */
#define CQ_DBL(p, y, dstrd)    \
    ((volatile void*) (((volatile unsigned char*) (p)) + 0x1000 + ((2*(y) + 1) * (4 << (dstrd)))) )


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
enum admin_command_set
{
    ADMIN_DELETE_SUBMISSION_QUEUE   = (0x00 << 7) | (0x00 << 2) | 0x00,
    ADMIN_CREATE_SUBMISSION_QUEUE   = (0x00 << 7) | (0x00 << 2) | 0x01,
    ADMIN_DELETE_COMPLETION_QUEUE   = (0x00 << 7) | (0x01 << 2) | 0x00,
    ADMIN_CREATE_COMPLETION_QUEUE   = (0x00 << 7) | (0x01 << 2) | 0x01,
    ADMIN_IDENTIFY_CONTROLLER       = (0x00 << 7) | (0x01 << 2) | 0x02,
    ADMIN_ABORT                     = (0x00 << 7) | (0x02 << 2) | 0x00,
    ADMIN_SET_FEATURES              = (0x00 << 7) | (0x02 << 2) | 0x01,
    ADMIN_GET_FEATURES              = (0x00 << 7) | (0x02 << 2) | 0x02
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
    nvm_queue_t sq = controller->queues[0];
    nvm_queue_t cq = controller->queues[1];

    volatile uint32_t* aqa = AQA(register_ptr);
    *aqa = AQA$AQS(sq->max_entries - 1) | AQA$AQC(cq->max_entries - 1);

    volatile uint64_t* asq = ASQ(register_ptr);
    *asq = sq->page.bus_addr;

    volatile uint64_t* acq = ACQ(register_ptr);
    *acq = cq->page.bus_addr;
}


static int identify_controller(nvm_controller_t controller, volatile void* register_ptr)
{
    struct command* identify = sq_enqueue(controller->queues[0]);
    if (identify  == NULL)
    {
        return ENOSPC;
    }

    cmd_header(identify, ADMIN_IDENTIFY_CONTROLLER, 0);
    cmd_data_ptr(identify, NULL, controller->data, 1);
    identify->dword[10] = (0 << 16) | 0x01;

    struct command* get_features = sq_enqueue(controller->queues[0]);
    if (get_features == NULL)
    {
        return ENOSPC;
    }

    uint16_t expected_id = *CMD_CID(get_features);
    cmd_header(get_features, ADMIN_GET_FEATURES, 0);

    get_features->dword[10] = (0x03 << 8) | 0x07;
    get_features->dword[11] = 0;

    // Wait for completions
    sq_submit(controller->queues[0]);
    for (size_t i = 0; i < 2; ++i)
    {
        struct completion* cpl = cq_dequeue_block(controller->queues[1], controller);
        if (cpl == NULL)
        {
            fprintf(stderr, "Waiting for completion timed out\n");
            return ETIME;
        }

        if (SCT(cpl) != 0 && SC(cpl) != 0)
        {
            fprintf(stderr, "Command failed with status code type %x and status code %x\n", SCT(cpl), SC(cpl));
            return EIO;
        }

        if (*CPL_CID(cpl) == expected_id)
        {
            uint16_t max_queues = _MIN(cpl->dword[0] >> 16, cpl->dword[0] & 0xffff);
            controller->max_queues = _MIN(max_queues, controller->max_queues);
        }
    }

    cq_update(controller->queues[1]);

    // Extract information from identify structure
    unsigned char* bytes = controller->data->virt_addr;
    controller->max_data_size = bytes[77] * (1 << (12 + CAP$MPSMIN(register_ptr)));
    controller->sq_entry_size = _RB(bytes[512], 3, 0);
    controller->cq_entry_size = _RB(bytes[513], 3, 0);
    controller->max_out_cmds = *((uint16_t*) (bytes + 514));
    controller->n_ns = *((uint32_t*) (bytes + 516));

    return 0;
}


static int set_num_queues(nvm_controller_t controller)
{
    struct command* cmd = sq_enqueue(controller->queues[0]);
    if (cmd == NULL)
    {
        return ENOSPC;
    }

    cmd_header(cmd, ADMIN_SET_FEATURES, 0);

    cmd->dword[10] = (1 << 31) | 0x07;
    cmd->dword[11] = (controller->max_queues << 16) | controller->max_queues;

    sq_submit(controller->queues[0]);

    struct completion* cpl = cq_dequeue_block(controller->queues[1], controller);
    if (cpl == NULL)
    {
        return ETIME;
    }

    cq_update(controller->queues[1]);

    return 0;
}


int nvm_init(nvm_controller_t* handle, int fd, volatile void* register_ptr, size_t db_size)
{
    *handle = NULL;
    struct nvm_controller* ctrl;

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
    ctrl = malloc(sizeof(struct nvm_controller));
    if (ctrl == NULL)
    {
        fprintf(stderr, "Failed to allocate controller structure: %s\n", strerror(errno));
        return errno;
    }

    // Allocate buffer for controller data
    ctrl->data = get_ram_buffer(fd, page_size);
    if (ctrl->data == NULL)
    {
        fprintf(stderr, "Failed to allocate controller data memory\n");
        free(ctrl);
        return ENOMEM;
    }

    // Set controller properties
    ctrl->page_size = page_size;
    ctrl->dstrd = CAP$DSTRD(register_ptr);
    ctrl->timeout = CAP$TO(register_ptr) * 500UL;
    ctrl->max_data_size = 0;
    ctrl->max_entries = CAP$MQES(register_ptr) + 1;   // CAP.MQES is a 0's based value
    ctrl->cq_entry_size = sizeof(struct completion);
    ctrl->sq_entry_size = sizeof(struct command);
    ctrl->max_queues = (uint16_t) (db_size / (4 << ctrl->dstrd));
    ctrl->n_queues = 0;
    ctrl->queues = NULL;
    ctrl->n_ns = 0;
    ctrl->dbs = register_ptr;

    // Allocate queue handle table
    ctrl->queues = calloc(ctrl->max_queues, sizeof(struct nvme_queue*));
    if (ctrl->queues == NULL)
    {
        fprintf(stderr, "Failed to allocate queue handle table: %s\n", strerror(errno));
        nvm_free(ctrl, fd);
        return errno;
    }

    // Create admin submission/completion queue pair
    int err = nvm_prepare_queues(ctrl, NULL, NULL);
    if (err != 0)
    {
        fprintf(stderr, "Failed to allocate admin queue handles\n");
        nvm_free(ctrl, fd);
        return err;
    }
    
    err = get_ram_page(fd, &ctrl->queues[0]->page);
    if (err != 0)
    {
        fprintf(stderr, "Failed to allocate and pin admin queue memory\n");
        nvm_free(ctrl, fd);
        return ENOMEM;
    }
    ctrl->queues[0]->max_entries = _MIN(ctrl->max_entries, ctrl->queues[0]->page.page_size / ctrl->queues[0]->entry_size);

    err = get_ram_page(fd, &ctrl->queues[1]->page);
    if (err != 0)
    {
        fprintf(stderr, "Failed to allocate and pin admin queue memory\n");
        nvm_free(ctrl, fd);
        return ENOMEM;
    }
    ctrl->queues[1]->max_entries = _MIN(ctrl->max_entries, ctrl->queues[1]->page.page_size / ctrl->queues[1]->entry_size);

    memset(ctrl->queues[0]->page.virt_addr, 0, ctrl->queues[0]->page.page_size);
    memset(ctrl->queues[1]->page.virt_addr, 0, ctrl->queues[1]->page.page_size);

    // Reset controller
    reset_controller(register_ptr, ctrl->timeout);

    // Set admin CQ and SQ
    configure_admin_queues(register_ptr, ctrl);

    // Bring controller back up
    enable_controller(register_ptr, host_page_size, ctrl->timeout);

    // Submit identify controller command
    err = identify_controller(ctrl, register_ptr);
    if (err != 0)
    {
        fprintf(stderr, "Failed to submit command: %s\n", strerror(err));
        nvm_free(ctrl, fd);
        return err;
    }

    // Set CQES and SQES in CC
    configure_entry_sizes(register_ptr, ctrl);

    // Set features
    err = set_num_queues(ctrl);
    if (err != 0)
    {
        fprintf(stderr, "Failed to submit command: %s\n", strerror(err));
        nvm_free(ctrl, fd);
        return err;
    }

    *handle = ctrl;
    return 0;
}


void nvm_free(nvm_controller_t ctrl, int ioctl_fd)
{
    if (ctrl != NULL)
    {
        // TODO issue delete queue command for all active queue pairs

        put_ram_buffer(ioctl_fd, ctrl->data);

        if (ctrl->queues != NULL)
        {
            for (uint16_t i = 0; i < ctrl->n_queues; ++i)
            {
                nvm_queue_t queue = ctrl->queues[i];
                // TODO: handle gpu and ram memory stuff
                free(queue);
            }

            free(ctrl->queues);
        }

        free(ctrl);
    }
}


/* Helper function to clear queue's page struct */
static void clear_page(page_t* page)
{
    page->kernel_handle = -1;
    page->device = -1;
    page->virt_addr = NULL;
    page->page_size = 0;
    page->bus_addr = 0;
}


/* Helper function to clear state in a queue handle */
static void clear_queue_handle(nvm_queue_t queue, nvm_controller_t ctrl, uint16_t no)
{
    queue->no = no / 2;
    clear_page(&queue->page);
    queue->max_entries = 0;
    queue->entry_size = (no % 2 == 0) ? ctrl->sq_entry_size : ctrl->cq_entry_size;
    queue->head = 0;
    queue->tail = 0;
    queue->phase = 1;
    queue->db = (no % 2 == 0) ? SQ_DBL(ctrl->dbs, queue->no, ctrl->dstrd) : CQ_DBL(ctrl->dbs, queue->no, ctrl->dstrd);
}


int nvm_prepare_queues(nvm_controller_t ctrl, nvm_queue_t* cq_handle, nvm_queue_t* sq_handle)
{
    if ((ctrl->n_queues - 2) / 2 >= (ctrl->max_queues - 1))
    {
        fprintf(stderr, "Maximum number of queues already created\n");
        return ENOSPC;
    }

    struct nvm_queue* sq = malloc(sizeof(struct nvm_queue));
    if (sq == NULL)
    {
        fprintf(stderr, "Failed to allocate queue handle: %s\n", strerror(errno));
        return errno;
    }

    clear_queue_handle(sq, ctrl, ctrl->n_queues);

    struct nvm_queue* cq = malloc(sizeof(struct nvm_queue));
    if (cq == NULL)
    {
        fprintf(stderr, "Failed to allocate queue handle: %s\n", strerror(errno));
        return errno;
    }

    clear_queue_handle(cq, ctrl, ctrl->n_queues + 1);

    ctrl->n_queues += 2;
    ctrl->queues[ctrl->n_queues - 2] = sq;
    ctrl->queues[ctrl->n_queues - 1] = cq;

    if (cq_handle != NULL)
    {
        *cq_handle = cq;
    }

    if (sq_handle != NULL)
    {
        *sq_handle = sq;
    }

    return 0;
}


static int create_cq(nvm_controller_t ctrl, nvm_queue_t queue)
{
    struct command* cmd = sq_enqueue(ctrl->queues[0]);
    if (cmd == NULL)
    {
        return EAGAIN;
    }

    queue->max_entries = _MIN(ctrl->max_entries, queue->page.page_size / queue->entry_size);

    cmd_header(cmd, ADMIN_CREATE_COMPLETION_QUEUE, 0);

    cmd->dword[6] = (uint32_t) queue->page.bus_addr;
    cmd->dword[7] = (uint32_t) (queue->page.bus_addr >> 32);
    cmd->dword[8] = 0;
    cmd->dword[9] = 0;

    cmd->dword[10] = ((queue->max_entries - 1) << 16) | queue->no;
    cmd->dword[11] = (0x0000 << 16) | (0x00 << 1) | 0x01;

    sq_submit(ctrl->queues[0]);

    struct completion* cpl = cq_dequeue_block(ctrl->queues[1], ctrl);
    if (cpl == NULL)
    {
        return ETIME;
    }

    // TODO check status code
    return 0;
}


static int create_sq(nvm_controller_t ctrl, nvm_queue_t queue)
{
    struct command* cmd = sq_enqueue(ctrl->queues[0]);
    if (cmd == NULL)
    {
        return EAGAIN;
    }

    queue->max_entries = _MIN(ctrl->max_entries, queue->page.page_size / queue->entry_size);

    cmd_header(cmd, ADMIN_CREATE_SUBMISSION_QUEUE, 0);

    cmd->dword[6] = (uint32_t) queue->page.bus_addr;
    cmd->dword[7] = (uint32_t) (queue->page.bus_addr >> 32);
    cmd->dword[8] = 0;
    cmd->dword[9] = 0;
    
    cmd->dword[10] = ((queue->max_entries - 1) << 16) | queue->no;
    cmd->dword[11] = (((uint32_t) queue->no) << 16) | (0x00 << 1) | 0x01;

    sq_submit(ctrl->queues[0]);

    struct completion* cpl = cq_dequeue_block(ctrl->queues[1], ctrl);
    if (cpl == NULL)
    {
        return ETIME;
    }

    // TODO Check status code
    return 0;
}


int nvm_commit_queues(nvm_controller_t ctrl)
{
    int err;

    // Create IO queue pairs
    for (uint16_t i = 2; i < ctrl->n_queues; i += 2)
    {
        err = create_cq(ctrl, ctrl->queues[i + 1]);
        if (err != 0)
        {
            return err;
        }

        err = create_sq(ctrl, ctrl->queues[i]);
        if (err != 0)
        {
            return err;
        }
    }

    return 0;
}

