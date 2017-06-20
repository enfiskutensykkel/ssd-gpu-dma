#ifndef __MESSAGE_H__
#define __MESSAGE_H__

#include "nvm/command.h"
#include <stdint.h>


int send_completion(uint32_t node_id, uint32_t intno, const struct completion* cpl, uint32_t timeout);


int remote_command(uint32_t node_id, uint32_t intno, const struct command* cmd, struct completion* cpl, uint32_t timeout);


#endif
