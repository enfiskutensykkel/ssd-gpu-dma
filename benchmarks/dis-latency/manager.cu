#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <vector>
#include <condition_variable>
#include <cstring>
#include <cerrno>
#include <ctime>
#include <cstdlib>
#include <csignal>
#include <sys/time.h>
#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_manager.h>
#include <nvm_rpc.h>
#include <nvm_util.h>
#include <nvm_queue.h>
#include <nvm_command.h>
#include "settings.h"
#include "segment.h"
#include "transfer.h"
#include <sisci_api.h>

using std::cerr;
using std::cout;
using std::endl;
using std::runtime_error;



static uint16_t randomId()
{
    static uint16_t uniqueCounter = 0;

    timeval tv;
    srand(time(NULL));

    if (gettimeofday(&tv, NULL) == 0)
    {
        srand(tv.tv_usec);
    }

    return (rand() + uniqueCounter++) & 0xffff;
}


static uint64_t randomBlock(size_t max)
{
    return rand() & (max - 1);
}


static uint64_t currentTime()
{
    timespec ts;

    if (clock_gettime(CLOCK_REALTIME, &ts) < 0)
    {
        throw runtime_error(strerror(errno));
    }

    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}


static void transfer(nvm_queue_t& cq, nvm_queue_t& sq, nvm_ctrl_t controller, const Settings& settings, bool warmup)
{
    auto dataSegment(createSegment(randomId(), 128 * controller->page_size));
    auto prpListSegment(createSegment(randomId(), controller->page_size));

    DmaPtr dataMap(createDmaMapping(dataSegment, controller, settings.ctrlAdapter));
    DmaPtr prpMap(createDmaMapping(prpListSegment, controller, settings.ctrlAdapter));

    for (size_t size: settings.transferSizes)
    {
        for (size_t i = 0; i < settings.repetitions; ++i)
        {
            nvm_cmd_t* cmd = sq_enqueue(&sq);
            nvm_cmd_header(cmd, NVM_IO_READ, settings.nvmNamespace);

            setDataPointer(cmd, dataMap, prpMap, settings.blockSize, size);

            uint16_t numBlocks = (size / settings.blockSize) + (size % settings.blockSize != 0);

            uint64_t block = randomBlock(settings.maxSize); 
            cmd->dword[10] = block;
            cmd->dword[11] = block >> 32;
            cmd->dword[12] = numBlocks - 1;

            uint64_t before = currentTime();
            sq_submit(&sq);

            nvm_cpl_t* cpl = nullptr;
            while ((cpl = cq_dequeue(&cq)) == nullptr);

            uint64_t after = currentTime();

            if (!CPL_OK(cpl))
            {
                throw runtime_error("Command failed");
            }

            sq_update_unchecked(&sq);
            cq_update(&cq);

            if (!warmup)
            {
                cout << size << "\t" << (after - before) << "\t" << block << "\t" << numBlocks << endl;
            }
        }
    }
}



static void runManager(nvm_ctrl_t controller, nvm_rpc_t rpc, const Settings& settings)
{
    nvm_queue_t cq;
    nvm_queue_t sq;

    SegmentPtr ioQueueMem(createSegment(randomId(), controller->page_size * 2));
    DmaPtr queueWindow(createDmaMapping(ioQueueMem, controller, settings.ctrlAdapter));
    memset((*queueWindow)->vaddr, 0, (*queueWindow)->n_ioaddrs * (*queueWindow)->page_size);

    int nvmerr = nvm_rpc_set_num_queues(rpc, 1, 1);
    if (nvmerr != 0)
    {
        throw runtime_error("Failed to set number of queues");
    }

    nvmerr = nvm_rpc_cq_create(&cq, rpc, controller, 1, (*queueWindow)->vaddr, (*queueWindow)->ioaddrs[0]);
    if (nvmerr != 0)
    {
        throw runtime_error("Failed to create CQ");
    }

    nvmerr = nvm_rpc_sq_create(&sq, rpc, controller, &cq, 1, 
            DMA_VADDR((*queueWindow)->vaddr, (*queueWindow)->page_size, 1), (*queueWindow)->ioaddrs[1]);
    if (nvmerr != 0)
    {
        throw runtime_error("Failed to create SQ");
    }

    transfer(cq, sq, controller, settings, true);
    transfer(cq, sq, controller, settings, false);
}


int main()
{
    Settings settings;

    nvm_ctrl_t controller;
    sci_error_t err;

    SCIInitialize(0, &err);

    int nvmerr = nvm_dis_ctrl_init(&controller, settings.ctrlId, settings.ctrlAdapter);
    if (nvmerr != 0)
    {
        cerr << "Failed to get controller reference: " << std::strerror(nvmerr) << endl;
        return 1;
    }

    SegmentPtr queueMemory;
    DmaPtr mapping;
    try
    {
        queueMemory = createSegment(randomId(), 2 * controller->page_size);
        mapping = createDmaMapping(queueMemory, controller, settings.ctrlAdapter);
        memset((*mapping)->vaddr, 0, (*mapping)->n_ioaddrs * (*mapping)->page_size);
    }
    catch (const runtime_error& err)
    {
        nvm_ctrl_free(controller);
        cerr << err.what() << endl;
        return 1;
    }

    nvm_manager_t manager = nullptr;
    nvmerr = nvm_manager_register(&manager, controller, *mapping.get());
    if (nvmerr != 0)
    {
        nvm_ctrl_free(controller);
        cerr << "Failed to reset controller" << endl;
        return 1;
    }

    nvm_rpc_t rpcRef = nullptr;
    nvm_rpc_bind_local(&rpcRef, manager);
    try
    {
        runManager(controller, rpcRef, settings);
    }
    catch (const runtime_error& err)
    {
        nvm_rpc_unbind(rpcRef);
        nvm_manager_unregister(manager);
        nvm_ctrl_free(controller);

        cerr << err.what() << endl;
        return 1;
    }

    nvm_rpc_unbind(rpcRef);
    nvm_manager_unregister(manager);
    nvm_ctrl_free(controller);
    SCITerminate();
    return 0;
}

