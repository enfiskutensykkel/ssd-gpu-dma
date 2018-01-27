#include <nvm_error.h>
#include <nvm_util.h>
#include <stddef.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>


static const char* generic_status[] =
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



static const char* generic_status_nvm_commands[] = 
{
    "LBA out of range",
    "Capacity exceeded",
    "Namespace not ready",
    "Reservation conflict",
    "Format in progress"
};



static const char* command_specific_status[] = 
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



static const char* command_specific_status_nvm_commands[] =
{
    "Conflicting attributes",
    "Invalid protection information",
    "Attempted write to read only range"
};



static const char* media_and_data_integrity_nvm_commands[] = 
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



static const char* lookup_string(uint8_t status_code_type, uint8_t status_code)
{
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



const char* nvm_strerror(int status)
{
    int err;
    uint8_t sct;
    uint8_t sc;
    
    err = NVM_ERR_UNPACK_ERRNO(status);
    sct = NVM_ERR_UNPACK_SCT(status);
    sc = NVM_ERR_UNPACK_SC(status);

    if (sct != 0 || sc != 0)
    {
        return lookup_string(sct, sc);
    }

    return strerror(err);
}

