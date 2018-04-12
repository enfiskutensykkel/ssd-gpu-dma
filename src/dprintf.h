#ifndef __NVM_INTERNAL_DPRINTF_H__
#define __NVM_INTERNAL_DPRINTF_H__

#ifndef NDEBUG

#include <nvm_util.h>
#include <nvm_error.h>
#include <stdio.h>
#include <stdarg.h>

/* Debug printf */
static void _nvm_dprintf(const char* func, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    fprintf(stderr, "[%s] ", func);
    vfprintf(stderr, format, args);
    va_end(args);
}

#define dprintf(...)            _nvm_dprintf(__func__, __VA_ARGS__)

#define _nvm_strerror(status)   nvm_strerror(status)
#define _SCIGetErrorString(err) SCIGetErrorString(err)

#endif /* ! NDEBUG */



/* If no debug print, just swallow message */
#ifndef dprintf
#define dprintf(...)
#endif



/* If no debug print, don't lookup completions */
#ifndef _nvm_strerror
#define _nvm_strerror(status)
#define _SCIGetErrorString(err)
#endif


#endif /* __NVM_INTERNAL_DPRINTF_H__ */
