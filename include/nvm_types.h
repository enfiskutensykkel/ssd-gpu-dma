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


/* 
 * NVM controller handle
 *
 * This structure represents an NVM controller and holds information that 
 * can be extracted from reading controller registers.
 */
typedef struct nvm_controller
{
    size_t                  page_size;      // Memory page size used by the controller (MPS)
    uint8_t                 dstrd;          // Doorbell stride (in encoded form)
    uint64_t                timeout;        // Controller timeout in milliseconds (TO)
    uint16_t                max_entries;    // Maximum queue entries supported (MQES)
    size_t                  mm_size;        // Size of memory-mapped region
    volatile void*          mm_ptr;         // Memory-mapped pointer to BAR0 of the physical device
} * nvm_ctrl_t;


/* 
 * Controller manager handle.
 *
 * Handle to the controller manager. The controller manager is responsible
 * for administrating the controller's admin queues (ASQ and ACQ).
 */
struct nvm_manager;
typedef struct nvm_manager* nvm_manager_t;


/*
 * Manager reference handle.
 *
 * Handle to the RPC interface for accessing NVM admin queues controlled by
 * the controller manager.
 */
struct nvm_rpc_reference;
typedef struct nvm_rpc_reference* nvm_rpc_t;


/*
 * DMA window
 *
 * This structure describes a region of memory that is accessible for the
 * NVM controller using DMA. The API assumes a continuous virtual memory
 * address, but the physical pages do not need to be contiguous.
 *
 * Note: Only page-aligned addresses are supported in NVM Express
 */
typedef struct __align__(32) nvm_dma_window
{
    void*                   vaddr;          // Virtual address to start of region (NB! can be NULL)
    size_t                  page_size;      // Controller's page size (MPS)
    size_t                  n_ioaddrs;      // Number of MPS-sized pages
    uint64_t                ioaddrs[1];     // Physical/IO addresses of the memory pages
} __attribute__((aligned (32))) * nvm_dma_t;


/* NVM IO queue handle
 *
 * This structure represents an IO queue and holds information about queue
 * entries and size as well as containing a pointer to the doorbell register.
 */
typedef struct __align__(64) nvm_queue
{
    uint16_t                no;             // Queue number (must be unique per SQ/CQ pair)
    uint16_t                max_entries;    // Maximum number of queue entries supported
    uint16_t                entry_size;     // Queue entry size
    uint32_t                head;           // Queue's head pointer
    uint32_t                tail;           // Queue's tail pointer
    int16_t                 phase;          // Current phase bit
    uint32_t                last;           // Used internally to check db writes
    volatile uint32_t*      db;             // Pointer to doorbell register (write only)
    volatile void*          vaddr;          // Virtual address to start of queue memory
    uint64_t                ioaddr;         // Physical/IO address of the memory page
} __attribute__((aligned (64))) nvm_queue_t;



/* NVM completion data type (16 bytes) */
typedef struct __align__(16) nvm_completion
{
    uint32_t                dword[4];       // The name DWORD is chosen to reflect the specification
} __attribute__((aligned (16))) nvm_cpl_t;



/* NVM command data type (64 bytes) */
typedef struct __align__(64) nvm_command
{
    uint32_t                dword[16];
} __attribute__((aligned (64))) nvm_cmd_t;



/*
 * Controller information structure.
 */
typedef struct nvm_controller_info
{
    uint32_t                nvme_version;   // NVM Express version number
    size_t                  page_size;      // Memory page size used by the controller (MPS)
    size_t                  db_stride;      // Doorbell stride (DSTRD)
    uint64_t                timeout;        // Controller timeout in milliseconds (TO)
    int                     contiguous;     // Contiguous queues required (CQR)
    uint16_t                max_entries;    // Maximum queue entries supported (MQES)
    uint8_t                 pci_vendor[4];  // PCI vendor and subsystem vendor identifier
    char                    serial_no[20];  // Serial number (NB! not null terminated)
    char                    model_no[40];   // Model number (NB! not null terminated)
    char                    firmware[8];    // Firmware revision
    size_t                  max_transfer_size;  // Maximum data transfer size (MDTS)
    size_t                  cq_entry_size;  // CQ entry size (CQES)
    size_t                  sq_entry_size;  // SQ entry size (SQES)
    size_t                  max_out_cmds;   // Maximum outstanding commands (MAXCMD)
    size_t                  max_n_ns;       // Maximum number of namespaces (NN)
} nvm_ctrl_info_t;


/*
 * Namespace information structure.
 */
typedef struct nvm_namespace_info
{
    uint32_t                ns_id;          // Namespace identifier
    size_t                  size;           // Size in logical blocks (NSZE)
    size_t                  capacity;       // Capacity in logical blocks (NCAP)
    size_t                  utilization;    // Utilization in logical blocks (NUSE)
    size_t                  lba_data_size;  // Logical block size (LBADS)
    size_t                  metadata_size;  // Metadata size (MS)
} nvm_ns_info_t;



#ifndef __CUDACC__
#undef __align__
#endif

#ifdef __cplusplus
}
#endif
#endif /* __DIS_NVM_TYPES_H__ */
