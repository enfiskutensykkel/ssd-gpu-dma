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



/* All namespaces identifier */
#define NVM_CMD_NS_ALL                  0xffffffff


/* List of NVM IO command opcodes */
enum nvm_io_command_set
{
    NVM_IO_FLUSH                    = (0x00 << 7) | (0x00 << 2) | 0x00, // 00h
    NVM_IO_WRITE                    = (0x00 << 7) | (0x00 << 2) | 0x01, // 01h
    NVM_IO_READ                     = (0x00 << 7) | (0x00 << 2) | 0x02, // 02h
    NVM_IO_WRITE_ZEROES             = (0x00 << 7) | (0x02 << 2) | 0x00  // 08h
};



/* List of NVM admin command opcodes */
enum nvm_admin_command_set
{
    NVM_ADMIN_DELETE_SUBMISSION_QUEUE   = (0x00 << 7) | (0x00 << 2) | 0x00,
    NVM_ADMIN_CREATE_SUBMISSION_QUEUE   = (0x00 << 7) | (0x00 << 2) | 0x01,
    NVM_ADMIN_DELETE_COMPLETION_QUEUE   = (0x00 << 7) | (0x01 << 2) | 0x00,
    NVM_ADMIN_CREATE_COMPLETION_QUEUE   = (0x00 << 7) | (0x01 << 2) | 0x01,
    NVM_ADMIN_IDENTIFY                  = (0x00 << 7) | (0x01 << 2) | 0x02,
    NVM_ADMIN_GET_LOG_PAGE              = (0x00 << 7) | (0x00 << 2) | 0x02,
    NVM_ADMIN_ABORT                     = (0x00 << 7) | (0x02 << 2) | 0x00,
    NVM_ADMIN_SET_FEATURES              = (0x00 << 7) | (0x02 << 2) | 0x01,
    NVM_ADMIN_GET_FEATURES              = (0x00 << 7) | (0x02 << 2) | 0x02
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
 * Build a PRP list consisting of PRP entries.
 *
 * Populate a memory page with PRP entries required for a transfer.
 * Returns the number of PRP entries used. Number of pages should 
 * always be max_data_size (MDTS) for IO commands.
 *
 * Note: currently, PRP lists can only be a single page
 */
__host__ __device__ static inline
size_t nvm_prp_list(size_t page_size, size_t n_pages, void* list_ptr, const uint64_t* data_ioaddrs)
{
    // TODO #ifdef __NO_COHERENCE__, make a nvm_prp_list_far variant that does not call nvm_cache_flush()
    size_t prps_per_page = page_size / sizeof(uint64_t);
    size_t i_prp;
    uint64_t* list;
    
    if (prps_per_page < n_pages)
    {
        n_pages = prps_per_page;
    }
    
    list = (uint64_t*) list_ptr;
    for (i_prp = 0; i_prp < n_pages; ++i_prp)
    {
        list[i_prp] = data_ioaddrs[i_prp];
    }

    nvm_cache_flush(list_ptr, sizeof(uint64_t) * i_prp);

    return i_prp;
}



/*
 * Helper function to build a PRP list and set a command's data pointer fields.
 */
__host__ __device__ static inline
size_t nvm_cmd_data(nvm_cmd_t* cmd, 
                    size_t page_size, 
                    size_t n_pages, 
                    void* list_ptr, 
                    uint64_t list_ioaddr, 
                    const uint64_t* data_ioaddrs)
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

    dptr0 = data_ioaddrs[prp++];

    if (n_pages > 2 && list_ptr != NULL)
    {
        dptr1 = list_ioaddr;
        prp += nvm_prp_list(page_size, n_pages - 1, list_ptr, &data_ioaddrs[prp]);
    }
    else if (n_pages >= 2)
    {
        dptr1 = data_ioaddrs[prp++];
    }
    
    nvm_cmd_data_ptr(cmd, dptr0, dptr1);
    return prp;
}



#ifndef __CUDACC__
#undef __device__
#undef __host__
#endif

#endif /* __NVM_COMMAND_H__ */
