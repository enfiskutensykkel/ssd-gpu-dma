#ifndef __DIS_NVM_TYPES_H__
#define __DIS_NVM_TYPES_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#ifndef __CUDACC__
#define __align__(x)
#endif

struct pci_dev;


/* NVM controller handle
 *
 * This structure represents an NVM controller and holds information that 
 * can be extracted from reading controller registers.
 */
struct nvm_controller
{
    struct pci_dev*         dev_ref;        // PCI device reference (can be NULL)
    size_t                  page_size;      // Memory page size used by the controller (MPS)
    uint8_t                 dstrd;          // Doorbell stride (in encoded form)
    uint64_t                timeout;        // Controller timeout in milliseconds
    size_t                  max_entries;    // Maximum queue entries supported (MQES)
    size_t                  mm_size;        // Size of memory-mapped region
    volatile void*          mm_ptr;         // Memory-mapped pointer to BAR0 of the physical device
};


/* Convenience type for a controller reference */
typedef struct nvm_controller nvm_ctrl_t;


/* IO queue handle
 *
 * This structure represents an IO queue and holds information about queue
 * entries and size as well as containing a pointer to the doorbell register.
 */
struct __align__(64) nvm_queue
{
    uint64_t                no;             // Queue number (must be unique per SQ/CQ pair)
    size_t                  max_entries;    // Maximum number of queue entries supported
    size_t                  entry_size;     // Queue entry size
    uint32_t                head;           // Queue's head pointer
    uint32_t                tail;           // Queue's tail pointer
    int16_t                 phase;          // Current phase bit
    void*                   vaddr;          // Virtual address to the first queue entry
    uint64_t                ioaddr;         // Physical/IO address of the queue (only 1 memory page is supported)
    volatile uint32_t*      db;             // Pointer to doorbell register (write only)
} __attribute__((aligned (64)));


/* Convenience type for queue handle */
typedef struct nvm_queue nvm_queue_t;


/* NVM completion data type (16 bytes) */
struct __align__(16) nvm_completion
{
    uint32_t                dword[4];       // The name DWORD is chosen from the spec
} __attribute__((aligned (16)));


/* Convenience type for completion struct */
typedef struct nvm_completion nvm_cpl_t;


/* NVM command data type (64 bytes) */
//struct __align__(64) nvm_command
//{
//    uint32_t                dword[16];
//} __attribute__((aligned (64)));
struct nvm_command
{
    uint32_t                dword[16];
};


/* Convenience type for command struct */
typedef struct nvm_command nvm_cmd_t;


/* NVM controller information
 *
 * This structure holds information about a controller.
 *
 * Note: Can only be retrieved by a local manager.
 */
struct nvm_controller_information
{
    uint8_t                 pci_vendor[4];  // PCI vendor and subsystem vendor identifier
    char                    serial_no[20];  // Serial number (NB! not null terminated)
    char                    model_no[40];   // Model number (NB! not null terminated)
    uint32_t                nvme_version;   // NVM Express version number
    size_t                  max_data_size;  // Maximum data transfer size (MDTS)
    size_t                  n_cqs;          // Number of CQs
    size_t                  max_cqs;        // Maximum number of CQs
    size_t                  n_sqs;          // Number of SQs
    size_t                  max_sqs;        // Maximum number of SQs
    size_t                  cq_entry_size;  // CQ entry size (CQES)
    size_t                  sq_entry_size;  // SQ entry size (SQES)
    size_t                  max_out_cmds;   // Maximum outstanding commands (MAXCMD)
    size_t                  n_ns;           // Number of namespaces
};


/* Convenience type for controller feature structure */
typedef struct nvm_controller_information nvm_ctrl_info_t;


#ifndef __CUDACC__
#undef __align__
#endif

#ifdef __cplusplus
}
#endif
#endif
