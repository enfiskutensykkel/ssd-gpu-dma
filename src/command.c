#include <nvm_util.h>
#include <nvm_types.h>
#include <nvm_command.h>
#include <stddef.h>
#include <stdint.h>
#include <errno.h>


static char* generic_status[] =
{
    "Success",
    "Invalid command opcode",
    "Invalid field in command",
    "Command ID conflict",
    "Data transfer error",
    "Commands aborted due to power loss notification",
    "Internal error",
    "Command abort requested",
    "Command aborted due to SQ deletion",
    "Command aborted due to failed fused command",
    "Command aborted due to missing fused command",
    "Invalid namespace or format",
    "Command sequence error",
    "Invalid SGL segment descriptor",
    "Invalid number of SQL descriptors",
    "Data SGL length invalid",
    "Metadata SGL length invalid",
    "SGL descriptor type invalid",
    "Invalid use of controller memory buffer",
    "PRP offset invalid",
    "Atomic write unit exceeded",
    "Operation denied",
    "SGL offset invalid",
    "Unknown/reserved",
    "Host identifier inconsistent format",
    "Keep alive timer expired",
    "Keep alive timer invalid",
    "Command aborted due to preempt and abort",
    "Sanitize failed",
    "Sanitize in progress",
    "SGL data block granularity invalid",
    "Command not supported for queue in CMB"
};


static char* generic_status_nvm_commands[] = 
{
    "LBA out of range",
    "Capacity exceeded",
    "Namespace not ready",
    "Reservation conflict",
    "Format in progress"
};


static char* command_specific_status[] = 
{
    "Completion queue invalid",
    "Invalid queue identifier",
    "Invalid queue size",
    "Abort command limit exceeded",
    "Unknown/reserved",
    "Asynchronous event request limit exceeded",
    "Invalid firmware slot",
    "Invalid firmware image",
    "Invalid interrupt vector",
    "Invalid log page",
    "Invalid format",
    "Firmware activation requires conventional reset",
    "Invalid queue deletion",
    "Feature identifier not saveable",
    "Feature not changeable",
    "Feature not namespace specific",
    "Firmware activation requires NVM subsystem reset",
    "Firmware activation requires reset",
    "Firmware activation requires maximum time violation",
    "Firmware activation prohibited",
    "Overlapping range",
    "Namespace insufficient capacity",
    "Namespace identifier unavailable",
    "Unknown/reserved",
    "Namespace already attached",
    "Namespace is private",
    "Namespace not attached",
    "Thin provisioning not supported",
    "Controller list invalid",
    "Device self-test in progress",
    "Boot partition write prohibited",
    "Invalid controller identifier",
    "Invalid secondary controller state",
    "Invalid number of controller resources",
    "Invalid resource identifier"
};


static char* command_specific_status_nvm_commands[] =
{
    "Conflicting attributes",
    "Invalid protection information",
    "Attempted write to read only range"
};


static char* media_and_data_integrity_nvm_commands[] = 
{
    "Write fault",
    "Unrecovered read error",
    "End-to-end guard check error",
    "End-to-end application tag check error",
    "End-to-end reference tag check error",
    "Compare failure",
    "Access denied",
    "Deallocated or unwritten logical block"
};


const char* nvm_status(const nvm_cpl_t* cpl)
{
    uint8_t status_code_type = SCT(cpl);
    uint8_t status_code = SC(cpl);

    switch (status_code_type)
    {
        case 0x00: // Generic command status
            if (status_code < 0x20)
            {
                return generic_status[status_code];
            }
            else if (0x80 <= status_code && status_code <= 0x84)
            {
                return generic_status_nvm_commands[status_code - 0x80];
            }
            return "Unknown generic error";

        case 0x01: // Command specific status
            if (status_code < 0x23)
            {
                return command_specific_status[status_code];
            }
            else if (0x80 <= status_code && status_code <= 0x82)
            {
                return command_specific_status_nvm_commands[status_code - 0x80];
            }
            return "Unknown command specific error";

        case 0x02: // Media and data integrity errors
            if (0x80 <= status_code && status_code <= 0x87)
            {
                return media_and_data_integrity_nvm_commands[status_code - 0x80];
            }
            return "Unknown media or data integrity error";

        default:
            return "Unknown status code type";
    }
}


size_t nvm_prp_list(void* vaddr, size_t page_size, size_t size, const uint64_t* lists, const uint64_t* prps)
{
    size_t prps_per_page = page_size / sizeof(uint64_t);
    size_t n_prps = DMA_SIZE(size, page_size) / page_size;

    size_t i_list = 0;
    size_t i_prp = 0;
    size_t pos = 0;

    uint64_t* entries = (uint64_t*) vaddr;

    while (i_prp < n_prps)
    {
        if ((pos + 1) % prps_per_page == 0)
        {
            entries[pos++] = lists[++i_list];
        }

        entries[pos++] = prps[i_prp++];
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

