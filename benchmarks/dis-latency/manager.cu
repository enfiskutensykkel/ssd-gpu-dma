#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cstring>
#include <cerrno>
#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_manager.h>
#include "settings.h"
#include "segment.h"
#include <sisci_api.h>

using std::cerr;
using std::endl;


Settings::Settings()
{
    cudaDevice = -1;
    //controllerId = 0;
    controllerId = 0x10000;
    disAdapter = 0;
    nvmNamespace = 1;
}


static void runManager(nvm_ctrl_t controller, const Settings& settings)
{
    SegmentPtr aqSegment(createSegment(randomId(), 2 * controller->page_size));

//
//    SegmentPtr adminQueueSegment;
//    DmaPtr adminQueues;
//        adminQueueSegment = createSegment(0, 2 * controller->page_size);
//
//        adminQueues = createDmaMapping(adminQueueSegment, controller, settings.disAdapter);

}


int main()
{
    Settings settings;

    nvm_ctrl_t controller;
    sci_error_t err;

    SCIInitialize(0, &err);

    int nvmerr = nvm_dis_ctrl_init(&controller, settings.controllerId, settings.disAdapter);
    if (nvmerr != 0)
    {
        cerr << "Failed to get controller reference: " << std::strerror(nvmerr) << endl;
        return 1;
    }

    try
    {
    }
    catch (const std::runtime_error& err)
    {
        nvm_ctrl_free(controller);
        cerr << err.what() << endl;
        return 1;
    }

    nvm_ctrl_free(controller);
    SCITerminate();
    return 0;
}

