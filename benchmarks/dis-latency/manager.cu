#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <thread>
#include <mutex>
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
#include "settings.h"
#include "segment.h"
#include <sisci_api.h>

using std::cerr;
using std::endl;


static std::mutex mtx;
static std::condition_variable cv;
static bool stopRunning = false;


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


static void stopManager()
{
    std::lock_guard<std::mutex> lock(mtx);
    stopRunning = true;
    cv.notify_all();
}


static void runManager(nvm_ctrl_t controller, const ManagerSettings& settings)
{
    SegmentPtr targetBuffer(createSegment(randomId(), controller->page_size, settings.cudaDevice));

    DmaPtr targetWindow(createDmaMapping(targetBuffer, controller, settings.ctrlAdapter));

    for (uint32_t adapter: settings.adapters)
    {
        exportSegment(targetBuffer, adapter);
    }

    cerr << std::hex << (*targetWindow)->ioaddrs[0] << endl;
    
    std::unique_lock<std::mutex> lock(mtx);
    auto old = signal(SIGINT, (void (*)(int)) stopManager);

    cv.wait(lock, [] { return stopRunning; });

    signal(SIGINT, old);
    lock.unlock();
}


int main()
{
    ManagerSettings settings;

    nvm_ctrl_t controller;
    sci_error_t err;

    SCIInitialize(0, &err);

    int nvmerr = nvm_dis_ctrl_init(&controller, settings.controllerId, settings.ctrlAdapter);
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
    }
    catch (const std::runtime_error& err)
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

    try
    {
        runManager(controller, settings);
    }
    catch (const std::runtime_error& err)
    {
        nvm_manager_unregister(manager);
        nvm_ctrl_free(controller);

        cerr << err.what() << endl;
        return 1;
    }

    nvm_manager_unregister(manager);
    nvm_ctrl_free(controller);
    SCITerminate();
    return 0;
}

