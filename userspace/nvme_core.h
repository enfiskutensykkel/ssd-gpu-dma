#ifndef __NVME_CORE_H__
#define __NVME_CORE_H__

#include "nvme.h"

// TODO: commands and stuff



int create_queue(nvm_controller_t controller, int ioctl_fd, int device);



// TODO: Do I need to pass ioctl, or are page-locked pages free'd when free'd?
void delete_queue(nvm_controller_t controller, int queue_no);


#endif
