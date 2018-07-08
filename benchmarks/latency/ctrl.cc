#include "ctrl.h"
#include "buffer.h"
#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_aq.h>
#include <nvm_admin.h>
#include <nvm_util.h>
#include <nvm_error.h>
#include <nvm_rpc.h>
#include <string>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>

using error = std::runtime_error;
using std::string;


#ifdef __DIS_CLUSTER__
static void createAdminQueues(Controller& ctrl, uint16_t numQueues)
{
    // Create admin queue reference
    int status = nvm_aq_create(&ctrl.aq_ref, ctrl.ctrl, ctrl.aq_mem.get());
    if (!nvm_ok(status))
    {
        throw error(string("Failed to reset controller: ") + nvm_strerror(status));
    }

    // Request number of queues
    uint16_t n_cqs = numQueues;
    uint16_t n_sqs = numQueues;
    status = nvm_admin_request_num_queues(ctrl.aq_ref, &n_cqs, &n_sqs);
    if (!nvm_ok(status))
    {
        throw error(nvm_strerror(status));
    }

    n_cqs = std::min(n_cqs, numQueues);
    n_sqs = std::min(n_sqs, numQueues);
    ctrl.numQueues = std::min(n_cqs, n_sqs);
}
#endif



#ifdef __DIS_CLUSTER__
static void bindAdminQueues(Controller& ctrl, uint32_t adapter, uint16_t numQueues)
{
    int status = nvm_dis_rpc_bind(&ctrl.aq_ref, ctrl.ctrl, adapter);
    if (!nvm_ok(status)) 
    {
        throw error("Failed to get RPC reference: " + string(nvm_strerror(status)));
    }

    uint16_t n_cqs = numQueues;
    uint16_t n_sqs = numQueues;
    status = nvm_admin_get_num_queues(ctrl.aq_ref, &n_cqs, &n_sqs);
    if (!nvm_ok(status))
    {
        throw error(nvm_strerror(status));
    }

    n_cqs = std::min(n_cqs, numQueues);
    n_sqs = std::min(n_sqs, numQueues);
    ctrl.numQueues = std::min(n_cqs, n_sqs);
}
#endif



#ifdef __DIS_CLUSTER__
Controller::Controller(uint64_t ctrlId, uint32_t adapter, uint32_t segmentId, uint32_t nsId, uint16_t numQueues, bool manager)
    : ctrl(nullptr)
    , aq_ref(nullptr)
    , fd(-1)
{
    // Get controller reference
    int status = nvm_dis_ctrl_init(&ctrl, ctrlId, adapter);
    if (!nvm_ok(status))
    {
        throw error(string("Failed to get controller reference: ") + nvm_strerror(status));
    }

    // Figure out if we should create admin queues or use existing ones
    if (manager)
    {
        aq_mem = createHostDma(ctrl, ctrl->page_size * 3, adapter, segmentId);
        createAdminQueues(*this, numQueues);
    }
    else
    {
        aq_mem = createHostDma(ctrl, ctrl->page_size, adapter, segmentId);
        bindAdminQueues(*this, adapter, numQueues);
    }

    // Identify controller
    status = nvm_admin_ctrl_info(aq_ref, &info, NVM_DMA_OFFSET(aq_mem, aq_mem->n_ioaddrs-1), aq_mem->ioaddrs[aq_mem->n_ioaddrs-1]);
    if (!nvm_ok(status))
    {
        throw error(nvm_strerror(status));
    }

    // Identify namespace
    status = nvm_admin_ns_info(aq_ref, &ns, nsId, NVM_DMA_OFFSET(aq_mem, aq_mem->n_ioaddrs-1), aq_mem->ioaddrs[aq_mem->n_ioaddrs-1]);
    if (!nvm_ok(status))
    {
        throw error(nvm_strerror(status));
    }
}
#endif



Controller::Controller(const char* path, uint32_t ns_id, uint16_t numQueues)
    : ctrl(nullptr)
    , aq_ref(nullptr)
    , fd(-1)
{
    fd = open(path, O_RDWR | O_NONBLOCK);
    if (fd < 0)
    {
        throw error(string("Failed to open descriptor: ") + strerror(errno));
    }

    // Get controller reference
    int status = nvm_ctrl_init(&ctrl, fd);
    if (!nvm_ok(status))
    {
        throw error(string("Failed to get controller reference: ") + nvm_strerror(status));
    }

    // Create admin queue memory
    aq_mem = createHostDma(ctrl, ctrl->page_size * 3);
    
    // Create admin queue reference
    status = nvm_aq_create(&aq_ref, ctrl, aq_mem.get());
    if (!nvm_ok(status))
    {
        throw error(string("Failed to reset controller: ") + nvm_strerror(status));
    }

    // Identify controller
    status = nvm_admin_ctrl_info(aq_ref, &info, NVM_DMA_OFFSET(aq_mem, 2), aq_mem->ioaddrs[2]);
    if (!nvm_ok(status))
    {
        throw error(nvm_strerror(status));
    }

    // Identify namespace
    status = nvm_admin_ns_info(aq_ref, &ns, ns_id, NVM_DMA_OFFSET(aq_mem, 2), aq_mem->ioaddrs[2]);
    if (!nvm_ok(status))
    {
        throw error(nvm_strerror(status));
    }

    // Request number of queues
    uint16_t n_cqs = numQueues;
    uint16_t n_sqs = numQueues;
    status = nvm_admin_request_num_queues(aq_ref, &n_cqs, &n_sqs);
    if (!nvm_ok(status))
    {
        throw error(nvm_strerror(status));
    }

    n_cqs = std::min(n_cqs, numQueues);
    n_sqs = std::min(n_sqs, numQueues);
    this->numQueues = std::min(n_cqs, n_sqs);
}



Controller::~Controller()
{
    nvm_aq_destroy(aq_ref);
    nvm_ctrl_free(ctrl);
    if (fd > 0)
    {
        close(fd);
    }
}

