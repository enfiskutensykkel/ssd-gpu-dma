#ifndef __DIS_NVM_EXAMPLES_NODE_H__
#define __DIS_NVM_EXAMPLES_NODE_H__
#ifdef __DIS_CLUSTER__

#include <stdint.h>


int get_node_info(const char* adapter_name_or_node_id, uint32_t* node_id, uint32_t* adapter);


int get_local_node_id(uint32_t adapter, uint32_t* node_id);


#endif // __DIS_CLUSTER__
#endif // __DIS_NVM_EXAMPLES_NODE_H__
