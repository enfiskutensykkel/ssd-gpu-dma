#ifndef __NVM_ERROR_H__
#define __NVM_ERROR_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <nvm_types.h>
#include <nvm_util.h>




/* Get the status code type of an NVM completion. */
#define NVM_ERR_SCT(cpl)            ((uint8_t) _RB(*NVM_CPL_STATUS(cpl), 11, 9))



/* Get the status code of an NVM completion */
#define NVM_ERR_SC(cpl)             ((uint8_t) _RB(*NVM_CPL_STATUS(cpl), 8, 1))



/* Is do not retry flag set? */
#define NVM_ERR_DNR(cpl)            ((uint8_t) _RB(*NVM_CPL_STATUS(cpl), 15, 15))



/* Extract value from status field from NVM completion */
#define NVM_ERR_STATUS(cpl)         \
    ((int) ( (cpl) != NULL ? -((NVM_ERR_SCT(cpl) << 8) | NVM_ERR_SC(cpl)) : 0 ))



/* Convenience macro for checking if an NVM completion indicates success. */
#define NVM_ERR_OK(cpl)             ( !NVM_ERR_SCT(cpl) && !NVM_ERR_SC(cpl) )



/* Pack errno and NVM completion status into a single status variable */
#define NVM_ERR_PACK(cpl, err)      \
    ((int) ( (err) != 0 ? (err) : NVM_ERR_STATUS(cpl) ) )



/* Extract values from packed status */
#define NVM_ERR_UNPACK_ERRNO(status)    ((status > 0) ? (status) : 0)
#define NVM_ERR_UNPACK_SCT(status)      ((status < 0) ? (((-status) >> 8) & 0xff) : 0)
#define NVM_ERR_UNPACK_SC(status)       ((status < 0) ? ((-status) & 0xff) : 0)


/* Check if everything is okay */
#define nvm_ok(status)              ( !(status) )



/*
 * Get an error string associated with the status code type and status code.
 * This function calls strerror() if the packed status is a regular errno.
 */
const char* nvm_strerror(int status);


    
#ifdef __cplusplus
}
#endif
#endif /* __NVM_ERROR_H__ */
