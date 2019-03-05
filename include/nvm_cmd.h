#ifndef __NVM_COMMAND_H__
#define __NVM_COMMAND_H__

#ifndef __CUDACC__
#define __device__
#define __host__
#endif

#include <nvm_util.h>
#include <nvm_types.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>



/* All namespaces identifier */
#define NVM_CMD_NS_ALL                  0xffffffff
#define NVM_CMD_OPCODE(gen, fun, data)  (_WB((gen), 7, 7) | _WB((fun), 6, 2) | _WB((data), 1, 0))


/* List of NVM IO command opcodes */
enum nvm_io_command_set
{
    NVM_IO_FLUSH            = NVM_CMD_OPCODE(0, 0, 0),  // 00h
    NVM_IO_WRITE            = NVM_CMD_OPCODE(0, 0, 1),  // 01h
    NVM_IO_READ             = NVM_CMD_OPCODE(0, 0, 2),  // 02h
    NVM_IO_WRITE_ZEROES     = NVM_CMD_OPCODE(0, 2, 0)   // 08h
};



/* List of NVM admin command opcodes */
enum nvm_admin_command_set
{
    NVM_ADMIN_DELETE_SQ     = NVM_CMD_OPCODE(0, 0, 0),  // 00h
    NVM_ADMIN_CREATE_SQ     = NVM_CMD_OPCODE(0, 0, 1),  // 01h
    NVM_ADMIN_GET_LOG_PAGE  = NVM_CMD_OPCODE(0, 0, 2),  // 02h
    NVM_ADMIN_DELETE_CQ     = NVM_CMD_OPCODE(0, 1, 0),  // 04h
    NVM_ADMIN_CREATE_CQ     = NVM_CMD_OPCODE(0, 1, 1),  // 05h
    NVM_ADMIN_IDENTIFY      = NVM_CMD_OPCODE(0, 1, 2),  // 06h
    NVM_ADMIN_ABORT         = NVM_CMD_OPCODE(0, 2, 0),  // 08h
    NVM_ADMIN_SET_FEATURES  = NVM_CMD_OPCODE(0, 2, 1),  // 09h
    NVM_ADMIN_GET_FEATURES  = NVM_CMD_OPCODE(0, 2, 2)   // 0Ah
};



/*
 * Set command's DWORD0 and DWORD1
 */
__device__ __host__ static inline
void nvm_cmd_header(nvm_cmd_t* cmd, uint16_t cid, uint8_t opcode, uint32_t ns_id)
{
    cmd->dword[0] = ((uint32_t) cid << 16) | (0x00 << 14) | (0x00 << 8) | (opcode & 0x7f);
    cmd->dword[1] = ns_id;
}



/*
 * Set command's DPTR field (DWORD6-9)
 */
__device__ __host__ static inline
void nvm_cmd_data_ptr(nvm_cmd_t* cmd, uint64_t prp1, uint64_t prp2)
{
    cmd->dword[0] &= ~( (0x03 << 14) | (0x03 << 8) );

    cmd->dword[6] = (uint32_t) prp1;
    cmd->dword[7] = (uint32_t) (prp1 >> 32UL);
    cmd->dword[8] = (uint32_t) prp2;
    cmd->dword[9] = (uint32_t) (prp2 >> 32UL);
}



/*
 * Set command's block fields (DWORD10-12)
 */
__device__ __host__ static inline
void nvm_cmd_rw_blks(nvm_cmd_t* cmd, uint64_t start_lba, uint16_t n_blks)
{
    cmd->dword[10] = start_lba;
    cmd->dword[11] = start_lba >> 32;
    cmd->dword[12] = (cmd->dword[12] & 0xffff0000) | ((n_blks - 1) & 0xffff);
}



/*
 * Set command's dataset management (DSM) field (DWORD13)
 */
//__device__ __host__ static inline
//void nvm_cmd_dataset(nvm_cmd_t* cmd, bool sequential, bool low_latency)
//{
//    cmd->dword[13] = 0; // not supported yet
//}




/*
 * Build PRP list consisting of PRP entries.
 *
 * Populate a memory page with PRP entries required for a transfer.
 * If the number of pages exceed the available number of entries, the last
 * entry will not be used (so that it can be used to point to the next list).
 * (See Chapter 4.4, Figure 14 in the NVMe specification).
 *
 * Returns the number of PRP entries used. 
 */
__host__ __device__ static inline
size_t nvm_prp_list(const nvm_prp_list_t* list, size_t n_pages, const uint64_t* ioaddrs)
{
    size_t n_prps = list->page_size / sizeof(uint64_t);
    size_t i_prp;
    volatile uint64_t* entries = (volatile uint64_t*) list->vaddr;

    // Do we need to reserve the last entry for the next list?
    if (n_pages > n_prps)
    {
        --n_prps;
    }
    else
    {
        n_prps = n_pages;
    }

    // Populate list
    for (i_prp = 0; i_prp < n_prps; ++i_prp)
    {
        entries[i_prp] = ioaddrs[i_prp];
    }

    // Flush list cache
    if (list->local)
    {
        nvm_cache_flush((void*) list->vaddr, sizeof(uint64_t) * i_prps);
    }
    else
    {
        nvm_wcb_flush();
    }

    return i_prp;
}



/*
 * Build chain of PRP lists.
 * Returns the total number of PRP entries.
 */
__host__ __device__ static inline
size_t nvm_prp_list_chain(size_t n_lists, const nvm_prp_list_t* lists, size_t n_pages, const uint64_t* ioaddrs)
{
    size_t i_list;
    size_t list_prps;
    size_t n_prps;

    if (n_lists == 0 || lists == NULL)
    {
        return 0;
    }

    list_prps = nvm_prp_list(&lists[0], n_pages, ioaddrs);

    for (i_list = 1, n_prps = list_prps; i_list < n_lists && n_prps < n_pages; ++i_list, n_prps += list_prps)
    {
        volatile uint64_t* next_list_ptr = ((volatile uint64_t*) lists[i_list - 1].vaddr) + list_prps;
        *next_list_ptr = lists[i_list].ioaddr;

        list_prps = nvm_prp_list(&lists[i_list], n_pages - n_prps, &ioaddrs[n_prps]);
    }

    return n_prps;
}



/*
 * Helper function to build a PRP list and set a command's data pointer fields.
 * Number of pages should always be max_data_size (MDTS) for IO commands.
 */
__host__ __device__ static inline
size_t nvm_cmd_data(nvm_cmd_t* cmd, size_t n_lists, const nvm_prp_list_t* lists, size_t n_pages, const uint64_t* ioaddrs)
{
    size_t prp = 0;
    uint64_t dptr0 = 0;
    uint64_t dptr1 = 0;

#if !defined( NDEBUG ) && !defined( __CUDA_ARCH__ )
    if (n_pages == 0)
    {
        return 0;
    }
#endif
    if (lists == NULL)
    {
        n_lists = 0;
    }

    dptr0 = ioaddrs[prp++];

    if (n_pages > 2 && n_lists != 0)
    {
        prp += nvm_prp_list_chain(n_lists, lists, n_pages - 1, &ioaddrs[prp]);
        dptr1 = lists[0].ioaddr;
    }
    else if (n_pages >= 2)
    {
        dptr1 = ioaddrs[prp++];
    }
    
    nvm_cmd_data_ptr(cmd, dptr0, dptr1);
    return prp;
}



#ifndef __CUDACC__
#undef __device__
#undef __host__
#endif

#endif /* __NVM_COMMAND_H__ */
