#include <cuda.h>
#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_dma.h>
#include <nvm_manager.h>
#include <nvm_rpc.h>
#include <nvm_util.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <getopt.h>


/*
 * Specify benchmark arguments.
 */
struct benchmark_config
{
    int             dev_id;
    uint64_t        ctrl_id;
    size_t          n_queues;
    uint32_t        ns_id;
    size_t          n_blks;
    size_t          start_lba;
};


void parse_arguments(int argc, char** argv, benchmark_config& settings);


static void launch_benchmark(nvm_ctrl_t ctrl, nvm_dma_t queue_window, benchmark_config& settings)
{
    nvm_manager_t manager;

    int status = nvm_manager_register(&manager, ctrl, queue_window);
    if (status != 0)
    {
        throw std::string("Register manager failed");
    }

    nvm_rpc_t manager_ref;
    nvm_rpc_bind_local(&manager_ref, manager);

    size_t info_page = queue_window->n_ioaddrs - 1;
    nvm_ns_info_t info;
    status = nvm_rpc_ns_info(&info, manager_ref, settings.ns_id,
            DMA_VADDR(queue_window->vaddr, queue_window->page_size, info_page), queue_window->ioaddrs[info_page]);

    if (status != 0)
    {
        nvm_rpc_unbind(manager_ref);
        nvm_manager_unregister(manager);
        throw std::string("Getting namespace information failed");
    }

    std::cerr << info.lba_data_size << std::endl;

    nvm_rpc_unbind(manager_ref);
    nvm_manager_unregister(manager);
}


int main(int argc, char** argv)
{
    benchmark_config settings;

    try
    {
        parse_arguments(argc, argv, settings);
    }
    catch (char err)
    {
        return err;
    }

    int status;
    nvm_ctrl_t ctrl;
    if ((status = nvm_ctrl_init(&ctrl, settings.ctrl_id)) != 0)
    {
        std::cerr << "Failed to get controller reference: " << std::strerror(status) << std::endl;
        return 1;
    }

    cudaError_t err = cudaSetDevice(settings.dev_id);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // This doesn't work with IOMMU enabled, I blame the nvidia driver...
    err = cudaHostRegister((void*) ctrl->mm_ptr, ctrl->mm_size, cudaHostRegisterIoMemory);
    if (err != cudaSuccess)
    {
        nvm_ctrl_free(ctrl);
        std::cerr << "Failed to map IO memory to UVA: " << cudaGetErrorString(err) << std::endl;
        return 2;
    }

    size_t queue_mem_size = (2 * settings.n_queues + 3) * ctrl->page_size;
    void* queue_mem;
    status = posix_memalign(&queue_mem, ctrl->page_size, queue_mem_size);
    if (status != 0)
    {
        cudaHostUnregister((void*) ctrl->mm_ptr);
        nvm_ctrl_free(ctrl);
        std::cerr << "Failed to create memory: " << std::strerror(status) << std::endl;
        return 3;
    }
    memset(queue_mem, 0, queue_mem_size);

    nvm_dma_t queue_dma;
    status = nvm_dma_window_host_map(&queue_dma, ctrl, queue_mem, queue_mem_size);
    if (status != 0)
    {
        free(queue_mem);
        cudaHostUnregister((void*) ctrl->mm_ptr);
        nvm_ctrl_free(ctrl);
        std::cerr << "Failed to map queue memory: " << std::strerror(status) << std::endl;
        return 3;
    }

    try
    {
        launch_benchmark(ctrl, queue_dma, settings);
    }
    catch (const std::string& errstr)
    {
        std::cerr << errstr << std::endl;
        status = 1;
    }

    nvm_dma_window_free(queue_dma);
    free(queue_mem);
    cudaHostUnregister((void*) ctrl->mm_ptr);
    nvm_ctrl_free(ctrl);
    return status;
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


void parse_arguments(int argc, char** argv, benchmark_config& settings)
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

    settings.ns_id = 1;
    settings.ctrl_id = 0;
    settings.dev_id = 0;
    settings.n_blks = 1000;
    settings.start_lba = 0;
    settings.n_queues = 1;

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
                settings.dev_id = strtoul(optarg, &end, 10);
                if (end == nullptr || *end != '\0' || settings.dev_id >= dev_count)
                {
                    std::cerr << "Invalid CUDA device: `" << optarg << "'" << std::endl;
                    throw 'd';
                }
                break;

            case 'c':
                settings.ctrl_id = strtoul(optarg, &end, 10);
                if (end == nullptr || *end != '\0')
                {
                    std::cerr << "Invalid NVMe controller: `" << optarg << "'" << std::endl;
                    throw 'c';
                }
                break;

            case 'n':
                settings.ns_id = strtoul(optarg, &end, 0);
                if (end == nullptr || *end != '\0' || settings.ns_id == 0 || settings.ns_id == 0xffffffff)
                {
                    std::cerr << "Invalid NVM namespace: `" << optarg << "'" << std::endl;
                    throw 'n';
                }
                break;

            case 'b':
                settings.n_blks = strtoul(optarg, &end, 0);
                if (end == nullptr || *end != '\0' || settings.n_blks == 0)
                {
                    std::cerr << "Invalid number of blocks: `" << optarg << "'" << std::endl;
                    throw 'b';
                }
                break;

            case 's':
                settings.start_lba = strtoul(optarg, &end, 0);
                if (end == nullptr || *end != '\0')
                {
                    std::cerr << "Invalid block number: `" << optarg << "'" << std::endl;
                    throw 's';
                }
                break;

            case 'q':
                settings.n_queues = strtoul(optarg, &end, 10);
                if (end == nullptr || *end != '\0' || settings.n_queues == 0 || settings.n_queues > 0xffff)
                {
                    std::cerr << "Invalid number of NVM IO queues: `" << optarg << "'" << std::endl;
                    throw 'q';
                }
                break;
        }
    }
}

