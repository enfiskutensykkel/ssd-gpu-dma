#ifndef __NVM_INTERNAL_DEVICE_H__
#define __NVM_INTERNAL_DEVICE_H__

#include <stdint.h>

struct nvm_controller;
struct nvm_device;


#ifdef _SISCI

#include <sisci_types.h>

/*
 * Reference to a device in the cluster.
 *
 * This structure is used to hold a SmartIO reference to the physical 
 * controller.
 */
struct nvm_device
{
    uint64_t                device_id;      // SISCI SmartIO unique device ID
    uint32_t                adapter;        // DIS adapter number
    sci_desc_t              sd;             // SISCI virtual device
    sci_device_t            device;         // SISCI SmartIO device handle
};

/*
 * Look up device reference from a controller handle.
 */
const struct nvm_device* _nvm_dev_from_ctrl(const struct nvm_controller* ctrl);


/* 
 * Acquire a device reference.
 */
int _nvm_dev_get(struct nvm_device* device, uint64_t device_id, uint32_t adapter);


/*
 * Release device reference.
 */
void _nvm_dev_put(struct nvm_device* device);


#endif /* _SISCI */


#endif /* __NVM_INTERNAL_DEVICE_H__ */
