#include "command.h"
#include "util.h"
#include <stddef.h>
#include <stdint.h>
#include <errno.h>


uint64_t build_prp_list(size_t page_size, void* prp_list, size_t n_prps, const uint64_t* list_addrs, const uint64_t* prp_addrs)
{
    uint64_t* list_ptr = (uint64_t*) prp_list;
    size_t list_pos = 0;
    size_t i_prp = 0;
    size_t next_list = 1;
    size_t n_offs = page_size / sizeof(uint64_t);

    while (i_prp < n_prps)
    {
        list_ptr[list_pos++] = prp_addrs[i_prp++];

        if (list_pos == n_offs)
        {
            list_ptr[list_pos++] = list_addrs[next_list++];
        }
    }

    return list_addrs[0];
}


void cmd_data_ptr(struct command* cmd, uint64_t prp1, uint64_t prp2)
{
    cmd->dword[0] &= ~( (0x03 << 14) | (0x03 << 8) );

    cmd->dword[6] = (uint32_t) prp1;
    cmd->dword[7] = (uint32_t) (prp1 >> 32);
    cmd->dword[8] = (uint32_t) prp2;
    cmd->dword[9] = (uint32_t) (prp2 >> 32);
}


void cmd_header(struct command* cmd, uint8_t opcode, uint32_t ns_id)
{
    cmd->dword[0] &= 0xffff0000;
    cmd->dword[0] |= (0x00 << 14) | (0x00 << 8) | (opcode & 0x7f);
    cmd->dword[1] = ns_id;
}

