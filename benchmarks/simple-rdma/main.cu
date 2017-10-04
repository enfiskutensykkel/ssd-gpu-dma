#include <cuda.h>
#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_manager.h>
#include <nvm_rpc.h>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <getopt.h>
#include <errno.h>
#include "settings.h"
#include "dma.h"
#include "queue.h"
#include "transfer.h"
#include "benchmark.h"
#include "report.h"
#include "stats.h"


static void showUsage(const std::string& str)
{
    fprintf(stderr, "Usage: %s --ctrl=<ctrl id> --namespace=<ns id>\n",
            str.c_str());
}


static void showHelp(const std::string& str)
{
    showUsage(str);
}


static void parseOptions(int argc, char** argv, Settings& settings)
{
    static option options[] = {
        { "help", no_argument, nullptr, 'h' },
        { "ctrl", required_argument, nullptr, 'c' },
        { "device", required_argument, nullptr, 'd' },
        { "namespace", required_argument, nullptr, 'n' },
        { "queues", required_argument, nullptr, 'q' },
        { "blocks", required_argument, nullptr, 'b' },
        { "start", required_argument, nullptr, 's' },
        { "repeat", required_argument, nullptr, 'r' },
        { "chunk", required_argument, nullptr, 't' },
        { nullptr, false, nullptr, 0 }
    };

    std::string optstr = ":";
    for (auto opt : options)
    {
        optstr += opt.val;
        if (opt.has_arg)
        {
            optstr += ":";
        }
    }

    // Set default settings
    settings.cudaDevice = 0;
    settings.controllerId = 0;
    settings.nvmNamespace = 1;
    settings.numQueues = 1;
    settings.numBlocks = 0x1000;
    settings.startBlock = 0;
    settings.repeatLoops = 1000;
    settings.chunkSize = 0; // Use the controller's MDTS
    settings.blockSize = 0; // Figure this out later

    // Figure out how many CUDA devices available
    int numDevs = 0;
    cudaError_t err = cudaGetDeviceCount(&numDevs);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get CUDA device count: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Parse options
    int optionsIdx = 0;
    char* endptr = nullptr;
    int option;
    while ((option = getopt_long(argc, argv, optstr.c_str(), options, &optionsIdx)) != -1)
    {
        switch (option)
        {
            case '?': // Unknown option
                fprintf(stderr, "Unknown option: `%s'\n", argv[optind - 1]);
                showUsage(argv[0]);
                exit(1);

            case ':': // Missing option argument
                fprintf(stderr, "Missing option argument: `%s'\n", argv[optind - 1]);
                showUsage(argv[0]);
                exit(1);

            case 'h': // Show help
                showHelp(argv[0]);
                exit(1);

            case 'c': // Set controller ID
                endptr = nullptr;
                settings.controllerId = strtoul(optarg, &endptr, 10);
                if (endptr == nullptr || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid NVM controller ID: `%s'\n", optarg);
                    exit(1);
                }
                break;

            case 'd': // Set CUDA device
                endptr = nullptr;
                settings.cudaDevice = strtol(optarg, &endptr, 10);
                if (endptr == nullptr || *endptr != '\0' || settings.cudaDevice < 0 || settings.cudaDevice >= numDevs)
                {
                    fprintf(stderr, "Invalid CUDA device: `%s'\n", optarg);
                    exit(1);
                }
                break;

            case 'n': // Set NVM namespace
                endptr = nullptr;
                settings.nvmNamespace = strtoul(optarg, &endptr, 10);
                if (endptr == nullptr || *endptr != '\0' || settings.nvmNamespace == 0)
                {
                    fprintf(stderr, "Invalid NVM namespace: `%s'\n", optarg);
                    exit(1);
                }
                break;

            case 'q': // Set number of submission queues
                endptr = nullptr;
                settings.numQueues = strtoul(optarg, &endptr, 0);
                if (endptr == nullptr || *endptr != '\0' || settings.numQueues == 0)
                {
                    fprintf(stderr, "Invalid number of SQs: `%s'\n", optarg);
                    exit(1);
                }
                break;

            case 'b': // Set number of blocks
                endptr = nullptr;
                settings.numBlocks = strtoul(optarg, &endptr, 0);
                if (endptr == nullptr || *endptr != '\0' || settings.numBlocks == 0)
                {
                    fprintf(stderr, "Invalid number of disk blocks: `%s'\n", optarg);
                    exit(1);
                }
                break;

            case 's': // Set start block
                endptr = nullptr;
                settings.startBlock = strtoul(optarg, &endptr, 0);
                if (endptr == nullptr || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid disk block: `%s'\n", optarg);
                    exit(1);
                }
                break;

            case 'r': // Set number of repeat loops for calculating average
                endptr = nullptr;
                settings.repeatLoops = strtoul(optarg, &endptr, 0);
                if (endptr == nullptr || *endptr != '\0' || settings.repeatLoops == 0)
                {
                    fprintf(stderr, "Invalid number of repeat loops: `%s'\n", optarg);
                    exit(1);
                }
                break;

            case 't': // Set chunk size
                endptr = nullptr;
                settings.chunkSize = strtoul(optarg, &endptr, 0);
                if (endptr == nullptr || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid maximum transfer size (chunk size): `%s'\n", optarg);
                    exit(1);
                }
                break;

            default:
                if (optionsIdx != 0)
                {
                    fprintf(stderr, "Option `--%s' not implemented!\n", options[optionsIdx].name);
                }
                else
                {
                    fprintf(stderr, "Option `-%c' not implemented!\n", option);
                }
                exit(2);
        }
    }
}


static void identify(nvm_rpc_t rpc, nvm_ctrl_t ctrl, Settings& settings)
{
    auto page_buffer = createHostBuffer(ctrl, std::max((size_t) 0x1000, ctrl->page_size));

    nvm_ctrl_info_t ci;
    int err = nvm_rpc_ctrl_info(&ci, rpc, ctrl, (*page_buffer)->vaddr, (*page_buffer)->ioaddrs[0]);
    if (err != 0)
    {
        throw std::runtime_error("Failed to identify controller");
    }

    nvm_ns_info_t ni;
    err = nvm_rpc_ns_info(&ni, rpc, settings.nvmNamespace, (*page_buffer)->vaddr, (*page_buffer)->ioaddrs[0]);
    if (err != 0)
    {
        throw std::runtime_error("Failed to identify namespace");
    }

    settings.chunkSize = std::min(ci.max_transfer_size, settings.chunkSize);
    if (settings.chunkSize == 0)
    {
        settings.chunkSize = ci.max_transfer_size;
    }
    settings.blockSize = ni.lba_data_size;
}


static void direct(nvm_ctrl_t controller, QueueList& queues, const Settings& settings, std::vector<uint64_t>& times)
{
    report("Creating device buffer");
    auto buffer(createDeviceBuffer(controller, settings.numBlocks * settings.blockSize, settings.cudaDevice));
    report(true);

    report("Preparing transfer descriptors directly to GPU");
    TransferList transfers;
    prepareTransfers(transfers, controller, queues, buffer, settings);
    report(true);

#if (!defined(NDEBUG) && defined(DEBUG))
    report("Verifying transfer descriptors");
    controlTransferMemory(transfers, buffer);
    report(true);
#endif

    report("Reading from disk directly to GPU");
    for (size_t i = 0; i < settings.repeatLoops; ++i)
    {
        uint64_t time = benchmark(queues, transfers);
        times.push_back(time);
    }
    report(true);
}


static void bounce(nvm_ctrl_t controller, QueueList& queues, const Settings& settings, std::vector<uint64_t>& ramTimes, std::vector<uint64_t>& gpuTimes)
{
    report("Creating host buffer");
    auto buffer(createHostBuffer(controller, settings.numBlocks * settings.blockSize));
    report(true);

    report("Creating device buffer");
    cudaError_t err = cudaSetDevice(settings.cudaDevice);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to set CUDA device");
    }

    void* devicePointer = nullptr;
    err = cudaMalloc(&devicePointer, settings.numBlocks * settings.blockSize);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate device buffer: " + std::string(cudaGetErrorString(err)));
    }
    report(true);

    report("Preparing transfer descriptors via RAM");
    TransferList transfers;
    prepareTransfers(transfers, controller, queues, buffer, settings);
    report(true);

#if (!defined(NDEBUG) && defined(DEBUG))
    report("Verifying transfer descriptors");
    controlTransferMemory(transfers, buffer);
    report(true);
#endif

    report("Reading from disk to GPU via RAM");
    for (size_t i = 0; i < settings.repeatLoops; ++i)
    {
        uint64_t time = benchmark(queues, transfers);

        uint64_t before = currentTime();
        err = cudaMemcpy(devicePointer, (*buffer)->vaddr, settings.numBlocks * settings.blockSize, cudaMemcpyHostToDevice);
        uint64_t after = currentTime();

        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to copy to device memory: " + std::string(cudaGetErrorString(err)));
        }

        ramTimes.push_back(time);
        time += after - before;
        gpuTimes.push_back(time);
    }
    report(true);
}


int main(int argc, char** argv)
{
    Settings settings;
    parseOptions(argc, argv, settings);

    report("Setting CUDA device");
    cudaError_t err = cudaSetDevice(settings.cudaDevice);
    report(err);
    if (err != cudaSuccess)
    {
        return 1;
    }

    // Create NVM controller reference
    nvm_ctrl_t controller;
    report("Getting controller reference");
    int nvmerr = nvm_ctrl_init(&controller, settings.controllerId);
    report(nvmerr);
    if (err != 0)
    {
        return 2;
    }

    // Allocate memory for ACQ and ASQ
    DmaPtr adminQueues;
    try
    {
        report("Creating admin queues");
        adminQueues = createHostBuffer(controller, 2 * controller->page_size);
        report(true);
    }
    catch (const std::runtime_error& err)
    {
        nvm_ctrl_free(controller);
        report(err);
        return 2;
    }

    // Reset NVM controller and configure admin queues
    nvm_manager_t manager = nullptr;
    report("Resetting controller");
    nvmerr = nvm_manager_register(&manager, controller, *adminQueues.get());
    report(nvmerr);
    if (nvmerr != 0)
    {
        nvm_ctrl_free(controller);
        return 2;
    }

    // Create RPC reference
    nvm_rpc_t rpcRef = nullptr;
    nvm_rpc_bind_local(&rpcRef, manager);

    // Identify controller and create IO queues
    QueueList queues;
    DmaPtr queueMemory;
    try
    {
        report("Identifying controller and namespace");
        identify(rpcRef, controller, settings);
        report(true);

        report("Creating IO queues");
        queueMemory = createHostBuffer(controller, (settings.numQueues + 1) * controller->page_size);
        
        createQueues(rpcRef, controller, queueMemory, queues);
        report(true);
    }
    catch (const std::runtime_error& err)
    {
        nvm_rpc_unbind(rpcRef);
        nvm_manager_unregister(manager);
        nvm_ctrl_free(controller);

        report(err);
        return 2;
    }

    // Run benchmark
    try
    {
        std::vector<uint64_t> gpuTimes;
        std::vector<uint64_t> ramTimes;

        bounce(controller, queues, settings, ramTimes, gpuTimes);

        showStatistics(settings, "RAM", ramTimes);
        printf("\n");

        showStatistics(settings, "Via RAM", gpuTimes);
        printf("\n");

        gpuTimes.clear();
        direct(controller, queues, settings, gpuTimes);
        showStatistics(settings, "GPU direct", gpuTimes);
    }
    catch (const std::runtime_error& err)
    {
        nvm_rpc_unbind(rpcRef);
        nvm_manager_unregister(manager);
        nvm_ctrl_free(controller);
        
        report(err);
        return 3;
    }
    
    // Release stuff and quit
    nvm_rpc_unbind(rpcRef);
    nvm_manager_unregister(manager);
    nvm_ctrl_free(controller);
    cudaDeviceReset(); // for cuda-memcheck
    return 0;
}

