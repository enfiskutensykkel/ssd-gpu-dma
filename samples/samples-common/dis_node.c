#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <errno.h>
#include <sisci_types.h>
#include <sisci_api.h>
#include <sisci_error.h>
#include "dis/node.h"
#include "util.h"


int get_node_info(const char* name, uint32_t* node_id, uint32_t* adapter)
{
    size_t i;
    sci_error_t err;
    uint64_t ul;

    if (parse_u64(name, &ul, 0) == 0)
    {
        *node_id = (uint32_t) ul;
        return 0;
    }

    uint32_t adapter_list[DIS_MAX_NSCIS];
    dis_nodeId_list_t node_id_list;
    dis_adapter_type_t adapter_type_list;

    char adapter_name[80];

    *node_id = 0;
    strncpy(adapter_name, name, sizeof(adapter_name));
    adapter_name[sizeof(adapter_name) - 1] = '\0';

    SCIGetNodeInfoByAdapterName(adapter_name, adapter_list, &node_id_list, &adapter_type_list, 0, &err);
    if (err != SCI_ERR_OK)
    {
        return EINVAL;
    }
    
    for (i = 0; i < DIS_MAX_NSCIS; ++i)
    {
        if (node_id_list[i] != 0)
        {
            *node_id = (uint32_t) node_id_list[i];
            *adapter = (uint32_t) adapter_list[i];
            return 0;
        }
    }
    
    return ENODEV;
}


int get_local_node_id(uint32_t adapter, uint32_t* node_id)
{
    sci_error_t err;
    uint32_t id = 0;

    SCIGetLocalNodeId(adapter, &id, 0, &err);
    if (err != SCI_ERR_OK)
    {
        return EIO;
    }

    *node_id = id;
    return 0;
}

