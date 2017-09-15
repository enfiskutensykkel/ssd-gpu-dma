#include <nvm_util.h>
#include <nvm_types.h>
#include <nvm_command.h>
#include <stddef.h>
#include <stdint.h>
#include <errno.h>


const char* nvm_strerror(const nvm_cpl_t* cpl)
{
    return "OK";
}


size_t nvm_prp_list(void* vaddr, size_t page_size, size_t transfer_size, const uint64_t* lists, const uint64_t* prps)
{
    size_t n_prps = DMA_SIZE(transfer_size, page_size) / page_size;

    size_t i_list = 0;
    size_t i_prp = 0;

    uint64_t* start_ptr = (uint64_t*) vaddr;
    uint64_t* entry_ptr = (uint64_t*) vaddr;

    while (i_prp < n_prps)
    {
        *entry_ptr++ = prps[i_prp++];

        if (((entry_ptr - start_ptr) & (page_size - 1)) == 0)
        {
            *entry_ptr++ = lists[++i_list];
        }
    }

    return i_prp;
}


size_t nvm_num_prp_pages(size_t page_size, size_t transfer_size)
{
    size_t prps_per_page = page_size / sizeof(uint64_t) - 1;
    size_t n_prps = DMA_SIZE(transfer_size, page_size) / page_size;
    size_t n_prp_pages = 1;

    while (n_prp_pages * prps_per_page + 1 < n_prps)
    {
        ++n_prp_pages;
    }

    return n_prp_pages;
}


void nvm_cmd_data_ptr(nvm_cmd_t* cmd, uint64_t prp1, uint64_t prp2)
{
    cmd->dword[0] &= ~( (0x03 << 14) | (0x03 << 8) );

    cmd->dword[6] = (uint32_t) prp1;
    cmd->dword[7] = (uint32_t) (prp1 >> 32);
    cmd->dword[8] = (uint32_t) prp2;
    cmd->dword[9] = (uint32_t) (prp2 >> 32);
}


void nvm_cmd_header(nvm_cmd_t* cmd, uint8_t opcode, uint32_t ns_id)
{
    cmd->dword[0] &= 0xffff0000;
    cmd->dword[0] |= (0x00 << 14) | (0x00 << 8) | (opcode & 0x7f);
    cmd->dword[1] = ns_id;
}

