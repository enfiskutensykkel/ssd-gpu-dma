#ifndef __DIS_NVM_INTERNAL_DPRINTF_H__
#define __DIS_NVM_INTERNAL_DPRINTF_H__

#ifndef NDEBUG

#include <stdio.h>
#include <stdarg.h>

#ifdef _SISCI
#include <sisci_error.h>
#include <sisci_api.h>
#endif


/* Debug printf */
static void _nvm_dprintf(const char* func, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    fprintf(stderr, "%s: ", func);
    vfprintf(stderr, format, args);
    va_end(args);
}

#define dprintf(...)    _nvm_dprintf(__func__, __VA_ARGS__)

#endif /* ! NDEBUG */

/* If no debug print, just swallow message */
#ifndef dprintf
#define dprintf(...)
#endif

#endif /* __DIS_NVM_INTERNAL_DPRINTF_H__ */
