#ifndef __DIS_NVM_INTERNAL_DPRINTNVM_H__
#define __DIS_NVM_INTERNAL_DPRINTNVM_H__

#ifndef NDEBUG

#include "dprintf.h"
#include <nvm_types.h>
#include <nvm_command.h>

/* Debug print failed command */
static void _nvm_dprintnvm(const char* func, const nvm_cmd_t* cmd, const nvm_cpl_t* cpl)
{
    if (cmd == NULL)
    {
        _nvm_dprintf(func, "NVM command failed: %s\n", nvm_strerror(cpl));
    }
    else
    {
        _nvm_dprintf(func, "NVM command %x failed: %s\n", cmd->dword[0] & 0x07, nvm_strerror(cpl));
    }
}

#define dprintnvm(cmd, cpl)   _nvm_dprintnvm(__func__, cmd, cpl)

#endif /* ! NDEBUG */

/* If no debug print, just swallow message */
#ifndef dprintnvm
#define dprintnvm(cmd, cpl)
#endif

#endif /* __DIS_NVM_INTERNAL_DPRINTNVM_H__ */
