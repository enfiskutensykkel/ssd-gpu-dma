#ifndef __NVM_TYPES_H__
#define __NVM_TYPES_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#ifndef __CUDACC__
#define __align__(x)
#endif



/* 
 * NVM controller handle.
 */
typedef struct
{
    size_t                  page_size;      // Memory page size used by the controller (MPS)
    uint8_t                 dstrd;          // Doorbell stride (in encoded form)
    uint64_t                timeout;        // Controller timeout in milliseconds (TO)
    uint16_t                max_entries;    // Maximum queue entries supported (MQES)
    size_t                  mm_size;        // Size of memory-mapped region
    volatile void*          mm_ptr;         // Memory-mapped pointer to BAR0 of the physical device
} nvm_ctrl_t;




/*
 * NVM admin queue-pair reference handle.
 *
 * As only a single process can be responsible of resetting the controller and
 * setting administration queues, this structure represents a remote handle to
 * that process. It is used as a descriptor for executing RPC calls to the 
 * remote process owning the admin queues.
 */
struct nvm_admin_reference;
typedef struct nvm_admin_reference* nvm_aq_ref;



/*
 * DMA mapping descriptor.
 *
 * This structure describes a region of memory that is accessible for the
 * NVM controller using DMA. The API assumes a continuous virtual memory
 * address, but the physical pages do not need to be contiguous.
 *
 * The structure contains a variably sized array of bus addresses that maps
 * to the physical memory pages. The user should therefore not create a local
 * instance of this descriptor, but rather rely on the API to allocate and
 * instantiate members.
 *
 * Note: Only page-aligned addresses are supported in NVM Express
 */
typedef struct __align__(32) 
{
    void*                   vaddr;          // Virtual address to start of region (NB! can be NULL)
    size_t                  page_size;      // Controller's page size (MPS)
    size_t                  n_ioaddrs;      // Number of MPS-sized pages
    uint64_t                ioaddrs[];      // Physical/IO addresses of the memory pages
} __attribute__((aligned (32))) nvm_dma_t;



/* 
 * NVM queue descriptor.
 *
 * This structure represents an NVM IO queue and holds information 
 * about memory addresses, queue entries as well as a memory mapped pointer to 
 * the device doorbell register. Maximum queue size is limited to a single 
 * page.
 *
 * Note: This descriptor represents both completion and submission queues.
 */
typedef struct __align__(64) 
{
    uint16_t                no;             // Queue number (must be unique per SQ/CQ pair)
    uint16_t                max_entries;    // Maximum number of queue entries supported
    uint16_t                entry_size;     // Queue entry size
    uint32_t                head;           // Queue's head pointer
    uint32_t                tail;           // Queue's tail pointer
    // TODO: Create bitfield for phase, add a remote field indicating
    //       if queue is far memory nor not, in which case we whould NOT do
    //       cache operations
    int16_t                 phase;          // Current phase bit
    uint32_t                last;           // Used internally to check db writes
    volatile uint32_t*      db;             // Pointer to doorbell register (NB! write only)
    volatile void*          vaddr;          // Virtual address to start of queue memory
    uint64_t                ioaddr;         // Physical/IO address of the memory page
} __attribute__((aligned (64))) nvm_queue_t;



/* 
 * NVM completion queue entry type (16 bytes) 
 */
typedef struct __align__(16) 
{
    uint32_t                dword[4];       // The name DWORD is chosen to reflect the specification
} __attribute__((aligned (16))) nvm_cpl_t;



/* 
 * NVM command queue entry type (64 bytes) 
 */
typedef struct __align__(64) 
{
    uint32_t                dword[16];
} __attribute__((aligned (64))) nvm_cmd_t;



/*
 * Controller information structure.
 *
 * Holds information about an NVM controller retrieved from reading on-board
 * registers and running an IDENTIFY CONTROLLER admin command.
 */
struct nvm_ctrl_info
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
    size_t                  max_data_size;  // Maximum data transfer size (MDTS)
    size_t                  max_data_pages; // Maximum data transfer size (in controller pages)
    size_t                  cq_entry_size;  // CQ entry size (CQES)
    size_t                  sq_entry_size;  // SQ entry size (SQES)
    size_t                  max_out_cmds;   // Maximum outstanding commands (MAXCMD)
    size_t                  max_n_ns;       // Maximum number of namespaces (NN)
};



/*
 * Namespace information structure.
 *
 * Holds informaiton about an NVM namespace.
 */
struct nvm_ns_info
{
    uint32_t                ns_id;          // Namespace identifier
    uint64_t                size;           // Size in logical blocks (NSZE)
    uint64_t                capacity;       // Capacity in logical blocks (NCAP)
    uint64_t                utilization;    // Utilization in logical blocks (NUSE)
    size_t                  lba_data_size;  // Logical block size (LBADS)
    size_t                  metadata_size;  // Metadata size (MS)
};



#ifndef __CUDACC__
#undef __align__
#endif

#ifdef __cplusplus
}
#endif
#endif /* __NVM_TYPES_H__ */
