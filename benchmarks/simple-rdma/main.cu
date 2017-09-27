#include <cuda.h>
#include <nvm_types.h>
#include <nvm_util.h>
#include <nvm_dma.h>
#include <nvm_rpc.h>
#include <nvm_command.h>
#include <memory>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <getopt.h>
#include "manager.h"
#include "queue.h"
#include "benchmark.h"


static void parse_arguments(int argc, char** argv, BenchmarkSettings& settings);


static void prepare_queues(ManagerPtr manager, uint16_t n_queues, QueueManagerPtr& queues)
{
    nvm_rpc_t ref;
    nvm_rpc_bind_local(&ref, manager->manager);

    uint16_t n_cqs = 1;
    uint16_t n_sqs = n_queues;

    int err = nvm_rpc_request_num_queues(ref, &n_cqs, &n_sqs);
    if (err != 0)
    {
        nvm_rpc_unbind(ref);
        throw std::runtime_error("Failed to set number of queues");
    }

    nvm_rpc_unbind(ref);

    n_queues = std::min(n_queues, n_sqs);

    queues = std::make_shared<QueueManager>(manager, 1, 1, n_queues);
}


static void print_line(std::ostream& os, const std::string& str)
{
    os << std::setfill('.');
    os << std::setw(40) << std::left;
    os << str;
    os << std::flush;
}


static void report_line(std::ostream& os, const std::string& title)
{
    os << std::setfill(' ');
    os << std::setw(20) << std::left;
    os << title + ":";
}


static void report_bw(std::ostream& os, const std::string& title, uint64_t usecs, size_t size)
{
    double bw = size / ((double) usecs);

    report_line(os, title);
    os << std::setprecision(3) << bw;
    os << " MiB/s";
    os << std::endl;
}


static void report_size(std::ostream& os, const std::string& title, size_t size)
{
    report_line(os, title);
    
    os << std::setprecision(3) << (size / ((float) (1UL << 20)));
    os << " MiB";
    os << std::endl;
}


static void launch_benchmark(ManagerPtr manager, QueueManagerPtr queues, const BenchmarkSettings& settings, DmaPtr prp_lists, DmaPtr ram, void* devptr, DmaPtr gpu)
{
    TupleList list;
    uint64_t bounce = 0;
    uint64_t direct = 0;

    print_line(std::cout, "Preparing PRP list");
    prepare_transfer_tuples(list, prp_lists, ram, settings);
    std::cout << "DONE" << std::endl;

    try
    {
        print_line(std::cout, "Reading from disk into GPU via RAM");
        bounce = time_transfer(list, manager->ctrl, queues, settings.ns, (*ram)->vaddr, devptr, settings.total_size, settings.dev_id);
        std::cout << "DONE" << std::endl;
    }
    catch (const std::runtime_error& err)
    {
        std::cout << "FAIL" << std::endl;
        throw err;
    }

    print_line(std::cout, "Preparing PRP list");
    prepare_transfer_tuples(list, prp_lists, gpu, settings);
    std::cout << "DONE" << std::endl;

    try
    {
        print_line(std::cout, "Reading from disk into GPU directly");
        direct = time_transfer(list, manager->ctrl, queues, settings.ns, nullptr, nullptr, settings.total_size, settings.dev_id);
        std::cout << "DONE" << std::endl;
    }
    catch (const std::runtime_error& err)
    {
        std::cout << "FAIL" << std::endl;
        throw err;
    }

    report_size(std::cout, "Transfer size", settings.total_size);
    report_bw(std::cout, "Bounced via RAM", bounce, settings.total_size);
    report_bw(std::cout, "Directly to GPU", direct, settings.total_size);
}

static DmaPtr create_host_buffer(nvm_ctrl_t ctrl, size_t page_size, size_t n_pages)
{
    void* ptr = nullptr;

    size_t size = page_size * n_pages;

    int err = posix_memalign(&ptr, ctrl->page_size, size);
    if (err != 0)
    {
        throw std::runtime_error("Failed to allocate RAM buffer");
    }

    nvm_dma_t* window = new nvm_dma_t;
    err = nvm_dma_window_host_map(window, ctrl, ptr, size);
    if (err != 0)
    {
        delete window;
        free(ptr);
        throw std::runtime_error("Failed to map RAM buffer");
    }

    DmaPtr dma_ptr(window, [ptr](nvm_dma_t* dma) {
        nvm_dma_window_free(*dma);
        free(ptr);
        delete dma;
    });

    return dma_ptr;
}


static DmaPtr create_device_buffer(nvm_ctrl_t ctrl, void* ptr, size_t page_size, size_t n_pages)
{
    size_t size = page_size * n_pages;

    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to get pointer attributes: " + std::string(cudaGetErrorString(err)));
    }

    nvm_dma_t* window = new nvm_dma_t;
    int status = nvm_dma_window_device_map(window, ctrl, attrs.devicePointer, size);
    if (status != 0)
    {
        delete window;
        throw std::runtime_error("Failed to map GPU buffer");
    }

    DmaPtr dma_ptr(window, [](nvm_dma_t* dma) {
        nvm_dma_window_free(*dma);
        delete dma;
    });

    return dma_ptr;
}


int main(int argc, char** argv)
{
    BenchmarkSettings settings;
    ManagerPtr manager;
    QueueManagerPtr queues;

    try
    {
        parse_arguments(argc, argv, settings);
    }
    catch (char err)
    {
        return err;
    }

    // Create controller reference
    try
    {
        print_line(std::cout, "Initializing NVMe controller");
        manager = std::make_shared<Manager>(settings.ctrl_id);
        std::cout << "DONE" << std::endl;

        print_line(std::cout, "Querying for block size");
        settings.blk_size = manager->get_blk_size(settings.ns);
        std::cout << "DONE" << std::endl;

        settings.page_size = manager->ctrl->page_size;
        settings.total_size = DMA_SIZE(settings.blk_size * settings.n_blks, settings.page_size);
        settings.chunk_size = std::min(manager->ctrl_info.max_transfer_size, settings.total_size);
        settings.n_prps = settings.total_size / settings.page_size;
        settings.n_prp_pages = nvm_num_prp_pages(settings.page_size, settings.chunk_size);
        settings.n_prp_lists = DMA_SIZE(settings.total_size, settings.chunk_size) / settings.chunk_size;
    }
    catch (const std::runtime_error& e)
    {
        std::cout << "FAIL" << std::endl;
        std::cerr << e.what() << std::endl;
        return 1;
    }

    cudaError_t err = cudaSetDevice(settings.dev_id);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // This doesn't work with IOMMU enabled, I blame the nvidia driver...
    err = cudaHostRegister((void*) manager->ctrl->mm_ptr, manager->ctrl->mm_size, cudaHostRegisterIoMemory);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to map IO memory to UVA: " << cudaGetErrorString(err) << std::endl;
        return 2;
    }

    // Create IO queues
    try
    {
        print_line(std::cout, "Preparing IO queues");
        prepare_queues(manager, settings.n_queues, queues);
        settings.n_queues = queues->submission_queues.size();
        std::cout << "DONE" << std::endl;
    }
    catch (const std::runtime_error& e)
    {
        std::cout << "FAIL" << std::endl;
        std::cerr << e.what() << std::endl;
        cudaHostUnregister((void*) manager->ctrl->mm_ptr);
        return 2;
    }

    // FIXME: Ugly bugly hack because I am lazy
    size_t host_page_size = 0x1000;
    size_t device_page_size = 0x10000;

    DmaPtr ram_buff;
    DmaPtr gpu_buff;
    DmaPtr prp_lists;

    void* devptr = nullptr;
    try
    {
        print_line(std::cout, "Allocating transfer buffers");

        cudaError_t err = cudaMalloc(&devptr, DMA_SIZE(settings.total_size, device_page_size));
        if (err != cudaSuccess)
        {
            throw std::runtime_error("Failed to allocate GPU buffer");
        }

        ram_buff = create_host_buffer(manager->ctrl, host_page_size, 
                DMA_SIZE(settings.total_size, host_page_size) / host_page_size);
        gpu_buff = create_device_buffer(manager->ctrl, devptr, device_page_size, 
                DMA_SIZE(settings.total_size, device_page_size) / device_page_size);

        std::cout << "DONE" << std::endl;

        print_line(std::cout, "Allocating PRP lists");
        prp_lists = create_host_buffer(manager->ctrl, settings.page_size, settings.n_prp_lists * settings.n_prp_pages);
        std::cout << "DONE" << std::endl;
    }
    catch (const std::runtime_error& e)
    {
        if (devptr != nullptr)
        {
            cudaFree(devptr);
        }

        std::cout << "FAIL" << std::endl;
        std::cerr << e.what() << std::endl;
        cudaHostUnregister((void*) manager->ctrl->mm_ptr);
        return 2;
    }

    try
    {
        launch_benchmark(manager, queues, settings, prp_lists, ram_buff, devptr, gpu_buff);
    }
    catch (const std::runtime_error& e)
    {
        cudaFree(devptr);
        cudaHostUnregister((void*) manager->ctrl->mm_ptr);
        std::cerr << e.what() << std::endl;
        return 1;
    }

    cudaFree(devptr);
    cudaHostUnregister((void*) manager->ctrl->mm_ptr);
    return 0;
}


static void give_usage(const std::string& name)
{
    std::cerr << "Usage: " << name << " [--dev=<cuda dev>] [--ctrl=<nvme dev>] [--namespace=<ns id>]"
       << " [--blocks=<num>] [--start=<blk>] [--queues <num>]" << std::endl;
}


static void show_help(const std::string& name)
{
    give_usage(name);
    std::cerr << std::endl;
    std::cerr << "    Benchmark transferring blocks to GPU memory." << std::endl << std::endl;
    std::cerr << "  --dev         <cuda dev>      CUDA device (default is 0)." << std::endl;
    std::cerr << "  --ctrl        <nvme dev>      NVMe disk identifier (default is 0)." << std::endl;
    std::cerr << "  --namespace   <ns id>         NVM namespace identifier (default is 1)." << std::endl;
    std::cerr << "  --blocks      <num>           Number of blocks (default is 1000)." << std::endl;
    std::cerr << "  --start       <blk>           Start at given block (default is 0)." << std::endl;
    std::cerr << "  --queues      <num>           Number of simultaneous queues to use (default 1)." << std::endl;
    std::cerr << std::endl;
}


static void parse_arguments(int argc, char** argv, BenchmarkSettings& cfg)
{
    static option opts[] = {
        { "help", no_argument, nullptr, 'h' },
        { "dev", required_argument, nullptr, 'd' },
        { "ctrl", required_argument, nullptr, 'c' },
        { "namespace", required_argument, nullptr, 'n' },
        { "blocks", required_argument, nullptr, 'b' },
        { "start", required_argument, nullptr, 's' },
        { "queues", required_argument, nullptr, 'q' },
        { nullptr, 0, nullptr, 0 }
    };

    int opt;
    int idx;
    char* end;

    int dev_count = 0;
    cudaError_t err = cudaGetDeviceCount(&dev_count);
    if (err != cudaSuccess)
    {
        std::cerr << "Something went wrong: " << cudaGetErrorString(err) << std::endl;
        throw 'd';
    }

    cfg.dev_id = 0;
    cfg.ctrl_id = 0;
    cfg.ns = 1;
    cfg.n_queues = 1;
    cfg.n_blks = 1000;
    cfg.blk_start = 0;

    while ((opt = getopt_long(argc, argv, ":hc:n:b:s:q:", opts, &idx)) != -1)
    {
        switch (opt)
        {
            case '?':
                std::cerr << "Unknown option: `" << argv[optind - 1] << "'" << std::endl;
                give_usage(argv[0]);
                throw '?';

            case ':':
                std::cerr << "Missing argument for option: `" << argv[optind - 1] << "'" << std::endl;
                give_usage(argv[0]);
                throw ':';

            case 'h':
                show_help(argv[0]);
                throw 0;

            case 'd':
                cfg.dev_id = strtoul(optarg, &end, 10);
                if (end == nullptr || *end != '\0' || cfg.dev_id >= dev_count)
                {
                    std::cerr << "Invalid CUDA device: `" << optarg << "'" << std::endl;
                    throw 'd';
                }
                break;

            case 'c':
                cfg.ctrl_id = strtoul(optarg, &end, 10);
                if (end == nullptr || *end != '\0')
                {
                    std::cerr << "Invalid NVMe controller: `" << optarg << "'" << std::endl;
                    throw 'c';
                }
                break;

            case 'n':
                cfg.ns = strtoul(optarg, &end, 0);
                if (end == nullptr || *end != '\0' || cfg.ns == 0 || cfg.ns == 0xffffffff)
                {
                    std::cerr << "Invalid NVM namespace: `" << optarg << "'" << std::endl;
                    throw 'n';
                }
                break;

            case 'b':
                cfg.n_blks = strtoul(optarg, &end, 0);
                if (end == nullptr || *end != '\0' || cfg.n_blks == 0)
                {
                    std::cerr << "Invalid number of blocks: `" << optarg << "'" << std::endl;
                    throw 'b';
                }
                break;

            case 's':
                cfg.blk_start = strtoul(optarg, &end, 0);
                if (end == nullptr || *end != '\0')
                {
                    std::cerr << "Invalid block number: `" << optarg << "'" << std::endl;
                    throw 's';
                }
                break;

            case 'q':
                cfg.n_queues = strtoul(optarg, &end, 10);
                if (end == nullptr || *end != '\0' || cfg.n_queues == 0 || cfg.n_queues > 0xffff)
                {
                    std::cerr << "Invalid number of NVM IO queues: `" << optarg << "'" << std::endl;
                    throw 'q';
                }
                break;
        }
    }
}

