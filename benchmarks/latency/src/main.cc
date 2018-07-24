#include <stdexcept>
#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include "ctrl.h"
#include "gpu.h"
#include "buffer.h"
#include "settings.h"
#include "transfer.h"
#include "utility.h"
#include "benchmark.h"
#include "stats.h"

#ifdef __DIS_CLUSTER__
#include <sisci_api.h>
#endif

using std::string;
using std::runtime_error;
using std::logic_error;
using std::make_shared;
using std::make_pair;
using std::to_string;



static QueuePtr createQueue(const CtrlPtr& ctrl, const QueueParam& qp, Settings& settings)
{
    uint16_t depth = qp.depth;
    if (depth == 0)
    {
        depth = ctrl->maxEntries - 1;
    }

    size_t chunk = qp.pages;
    if (chunk == 0)
    {
        if (settings.random)
        {
            chunk = 1;
        }
        else
        {
            chunk = ctrl->chunkSize / ctrl->pageSize;
        }
    }

    switch (qp.location)
    {
        case QueueLocation::LOCAL:
            return make_shared<LocalQueue>(ctrl, qp.no, depth, chunk, ctrl->adapter, settings.id());

        case QueueLocation::REMOTE:
            return make_shared<RemoteQueue>(ctrl, qp.no, depth, chunk, ctrl->adapter, settings.id());

        case QueueLocation::GPU:
            return make_shared<GpuQueue>(ctrl, qp.no, depth, chunk, 
                    make_shared<Gpu>(qp.fdid, qp.adapter), qp.adapter, settings.id(), settings.id());

        default:
            throw logic_error("Unknown queue location");
    }
}



static void prepareQueues(QueueMap& queues, const CtrlPtr& ctrl, Settings& settings)
{
    fprintf(stderr, "Preparing queues... ");
    if (settings.transferInfo)
    {
        fprintf(stderr, "\n");
    }

    for (const auto& qp: settings.queues)
    {
        if (queues.find(qp.no) != queues.end())
        {
            throw runtime_error("Queue number " + to_string(qp.no) + " is already used!");
        }

        auto queue = createQueue(ctrl, qp, settings);

        if (settings.transferInfo)
        {
            fprintf(stderr, "- QUEUE no=%02u location=%s cmds=%zu prps=%zu", queue->no, queue->type().c_str(), queue->depth, queue->pages);

            const GpuQueue* gpuQueue = dynamic_cast<const GpuQueue*>(queue.get());
            if (gpuQueue != nullptr)
            {
                const Gpu& gpu = gpuQueue->getGpu();
                fprintf(stderr, " (fdid=%lx bdf=%s name=%s)", gpu.fdid, gpu.deviceBdf().c_str(), gpu.deviceName().c_str());
            }
            fprintf(stderr, "\n");
        }

        queues.insert(make_pair(qp.no, queue));
    }

    if (!settings.transferInfo)
    {
        fprintf(stderr, "DONE\n");
    }
}



static size_t prepareChunks(ChunkList& chunks, const QueuePtr& queue, size_t pageOffset, uint64_t startBlock, size_t numBlocks)
{
    size_t totalPages = 0;
    uint64_t currBlock = 0;

    const auto& ctrl = queue->getController();

    const size_t blocksPerChunk = NVM_PAGE_TO_BLOCK(ctrl.pageSize, ctrl.blockSize, queue->pages);

    while (currBlock != numBlocks)
    {
        size_t blocks = std::min(numBlocks - currBlock, blocksPerChunk);
        size_t pages = NVM_PAGE_ALIGN(blocks * ctrl.blockSize, ctrl.pageSize) / ctrl.pageSize;
        size_t currPage = NVM_BLOCK_TO_PAGE(ctrl.pageSize, ctrl.blockSize, currBlock);

        Chunk chunk;
        chunk.startBlock = startBlock + currBlock;
        chunk.numBlocks = blocks;
        chunk.startPage = pageOffset + currPage;
        chunk.numPages = pages;

        chunks.push_back(chunk);
        currBlock += blocks;
        totalPages += pages;
    }

    return totalPages;
}



static MemoryPtr readFile(const string& filename, size_t size)
{
    fprintf(stderr, "Reading input file... ");
    MemoryPtr buffer(allocateHostMemory(size));

    FILE* fp = fopen(filename.c_str(), "rb");
    if (fp == nullptr)
    {
        throw runtime_error("Failed to open file for read: " + string(strerror(errno)));
    }

    size_t read = fread(buffer.get(), 1, size, fp);
    if (read < size) 
    {
        fprintf(stderr, "\nWARNING: Could only read %zu out of %zu bytes from file `%s'\n", read, size, filename.c_str());
    }
    else
    {
        fprintf(stderr, "DONE\n");
    }

    fclose(fp);

    return buffer;
}



static MemoryBufferPtr createBuffer(const Ctrl& ctrl, const GpuPtr& gpu, Settings& settings)
{
    size_t size = calculateBufferSize(ctrl, settings);

    MemoryBufferPtr buffer;
    if (gpu != nullptr)
    {
        buffer = make_shared<GpuBuffer>(gpu, ctrl, size, settings.id());
    }
    else
    {
        buffer = make_shared<MemoryBuffer>(ctrl, size, settings.id());
    }

    if (settings.transferInfo)
    {
        fprintf(stderr, "- BUFFER location=%s size=%zu prps=%zu blocks=%zu",
                gpu == nullptr ? "ram" : "gpu", buffer->size, buffer->buffer->n_ioaddrs, buffer->size / ctrl.blockSize);
        if (gpu != nullptr)
        {
            fprintf(stderr, " (device=%d fdid=%lx bdf=%s name=%s)", 
                    gpu->cudaDevice, gpu->fdid, gpu->deviceBdf().c_str(), gpu->deviceName().c_str());
        }
        fprintf(stderr, "\n");
    }

    return buffer;
}



static void prepareTransfers(TransferMap& transfers, const QueueMap& queues, const Ctrl& ctrl, const MemoryPtr& fileData, Settings& settings)
{
    size_t currentPage = 0;
    const size_t numBlocks = calculateTransferBlocks(ctrl, settings);
    const size_t totalPages = calculateBufferSize(ctrl, settings) / ctrl.pageSize;
    const size_t startBlock = calculateOffsetBlocks(ctrl, settings);

    if (startBlock + numBlocks > ctrl.namespaceSize)
    {
        throw runtime_error("Specified offset and length is larger than the size of the namespace");
    }

    fprintf(stderr, "Preparing buffers and transfer lists... ");

    if (settings.transferInfo)
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "- CTRL fdid=%lx namespace=%u total-blocks=%zu total-prps=%zu\n", ctrl.fdid, ctrl.namespaceId, numBlocks, totalPages);
    }

    // Take GPU reference 
    GpuPtr gpu;
    if (settings.gpu.fdid)
    {
        gpu = make_shared<Gpu>(settings.gpu.fdid, settings.gpu.adapter);
    }
    else if (settings.gpu.device != -1)
    {
        gpu = make_shared<Gpu>(settings.gpu.device);
    }

    if (gpu != nullptr && settings.transferInfo)
    {
        fprintf(stderr, "- GPU device=%d fdid=%lx bdf=%s name=%s\n", gpu->cudaDevice, gpu->fdid, gpu->deviceBdf().c_str(), gpu->deviceName().c_str());
    }

    // If sequential access, create a single buffer
    MemoryBufferPtr buffer;
    if (!settings.parallel || settings.shared)
    {
        buffer = createBuffer(ctrl, gpu, settings);
    }

    // Create queues and transfer lists
    for (auto it = queues.cbegin(); it != queues.cend(); ++it)
    {
        const auto& q = it->second;
        auto transfer = make_shared<Transfer>();
        transfer->queue = q;

        if (settings.parallel)
        {
            if (!settings.shared)
            {
                transfer->buffer = createBuffer(ctrl, gpu, settings);
            }
            else
            {
                transfer->buffer = buffer;
            }
            transfer->count = numBlocks;
            transfer->offset = startBlock;
            transfer->pageOffset = 0;
        }
        else
        {
            transfer->buffer = buffer;
            transfer->pageOffset = currentPage;
            auto prev = it++;

            if (it == queues.cend())
            {
                transfer->offset = startBlock + NVM_PAGE_TO_BLOCK(ctrl.pageSize, ctrl.blockSize, currentPage); 
                transfer->count = numBlocks - NVM_PAGE_TO_BLOCK(ctrl.pageSize, ctrl.blockSize, currentPage);
            }
            else
            {
                transfer->offset = startBlock + NVM_PAGE_TO_BLOCK(ctrl.pageSize, ctrl.blockSize, currentPage);
                transfer->count = NVM_PAGE_TO_BLOCK(ctrl.pageSize, ctrl.blockSize, totalPages / queues.size());
            }

            it = prev;
        }

        if (transfer->count == 0)
        {
            throw runtime_error("Not enough data for multiple queues, specify larger data count");
        }

        currentPage += prepareChunks(transfer->chunks, transfer->queue, transfer->pageOffset, transfer->offset, transfer->count);
        for (size_t i = 0; i < settings.innerIterations - 1; ++i)
        {
            prepareChunks(transfer->chunks, transfer->queue, transfer->pageOffset, transfer->offset, transfer->count);
        }

        // Sort transfer list randomly if specified
        if (settings.random)
        {
            std::shuffle(transfer->chunks.begin(), transfer->chunks.end(), std::default_random_engine());
        }

        if (settings.transferInfo)
        {
            fprintf(stderr, "- TRANSFER queue=%02u pattern=%s total-cmds=%zu total-blocks=%zu total-prps=%zu block-start=%zu block-count=%zu prp-start=%zu prp-count=%zu\n",
                    q->no, settings.random ? "random" : "sequential", 
                    transfer->chunks.size(), transfer->count * settings.innerIterations, NVM_BLOCK_TO_PAGE(ctrl.pageSize, ctrl.blockSize, transfer->count * settings.innerIterations),
                    transfer->offset, transfer->count, transfer->pageOffset, NVM_BLOCK_TO_PAGE(ctrl.pageSize, ctrl.blockSize, transfer->count));
        }

        // Load file data to buffer
        if (settings.write && fileData != nullptr)
        {
            size_t offset = transfer->pageOffset * ctrl.pageSize;
            const void* ptr = (void*) (((unsigned char*) fileData.get()) + offset);
            transfer->buffer->load(ptr, transfer->count * ctrl.blockSize, offset);
        }

        transfers.insert(make_pair(q->no, transfer));
    }

    if (!settings.transferInfo)
    {
        fprintf(stderr, "DONE\n");
    }
}



static void writeBuffer(const string& filename, const MemoryPtr& buffer, size_t size)
{
    FILE* fp = fopen(filename.c_str(), "wb");
    if (fp == nullptr)
    {
        throw runtime_error("Failed to open file for write: " + string(strerror(errno)));
    }

    size_t written = fwrite(buffer.get(), 1, size, fp);

    fclose(fp);
    if (written != size)
    {
        throw runtime_error("Failed to write " + to_string(size) + " bytes to file `" + filename + "'");
    }
}



static void writeFile(const TransferMap& transfers, const Ctrl& ctrl, const Settings& settings)
{
    size_t totalSize = calculateTransferBlocks(ctrl, settings) * ctrl.blockSize;
    MemoryPtr outputBuffer(allocateHostMemory(totalSize));

    fprintf(stderr, "Writing to output file... ");
    for (const auto& t: transfers)
    {
        uint16_t no = t.first;
        const auto& transfer = t.second;

        size_t size = transfer->count * ctrl.blockSize;
        size_t offset = transfer->pageOffset * ctrl.pageSize;

        void* output = (void*) (((unsigned char*) outputBuffer.get()) + offset);
        transfer->buffer->save(output, size, offset);

        if (settings.shared)
        {
            // All queues share the same buffer
            break;
        }

        if (settings.parallel)
        {
            string filename = settings.outputFilename + "-" + to_string(no);
            writeBuffer(filename, outputBuffer, size);
        }
    }

    if (!settings.parallel || settings.shared)
    {
        writeBuffer(settings.outputFilename, outputBuffer, totalSize);
    }

    fprintf(stderr, "DONE\n");
}



static void verifyTransfers(const MemoryPtr& fileContent, const TransferMap& transfers, const Ctrl& ctrl, const Settings& settings)
{
    size_t totalSize = calculateTransferBlocks(ctrl, settings) * ctrl.blockSize;
    MemoryPtr tempBuffer(allocateHostMemory(totalSize));

    fprintf(stderr, "Verifying buffers... ");
    for (const auto& t: transfers)
    {
        const auto& transfer = t.second;

        size_t size = transfer->count * ctrl.blockSize;
        size_t offset = transfer->pageOffset * ctrl.pageSize;

        void* output = (void*) (((unsigned char*) tempBuffer.get()) + offset);
        transfer->buffer->save(output, size, offset);

        if (settings.shared)
        {
            break;
        }

        if (settings.parallel)
        {
            if (memcmp(tempBuffer.get(), fileContent.get(), totalSize) != 0)
            {
                throw runtime_error("Memory buffer for queue differ from file content");
            }
        }
    }

    if (!settings.parallel || settings.shared)
    {
        if (memcmp(tempBuffer.get(), fileContent.get(), totalSize) != 0)
        {
            throw runtime_error("Memory buffer differ from file content");
        }
    }

    fprintf(stderr, "DONE\n");
}



int main(int argc, char** argv)
{
    Settings settings;

    try
    {
        settings.parseArguments(argc, argv);
    }
    catch (const string& e)
    {
        fprintf(stderr, "%s\n", e.c_str());
        return 1;
    }
    catch (const runtime_error& e)
    {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }

#ifdef __DIS_CLUSTER__
    sci_error_t err;
    SCIInitialize(0, &err);
#endif

    try
    {
        // Initialize controller manager and get controller reference
        CtrlManagerPtr manager;
        
        if (settings.ctrl.fdid != 0)
        {
            fprintf(stderr, "Taking controller reference (%s)... ",
                    settings.manager ? "local admin" : "remote admin");
            manager = make_shared<CtrlManager>(settings.ctrl.fdid, settings.ctrl.adapter, settings.id(), settings.manager, settings.nvmNamespace);
        }
        else
        {
            fprintf(stderr, "Resetting controller... ");
            manager = make_shared<CtrlManager>(settings.ctrl.path, settings.nvmNamespace);
        }
        fprintf(stderr, "DONE\n");

        const auto& ctrl = manager->getController();

        // Create queues
        QueueMap queues;
        prepareQueues(queues, manager->getControllerPtr(), settings);

        // Read data from file (if filename is specified)
        MemoryPtr fileData;
        if (!settings.inputFilename.empty())
        {
            const size_t numBlocks = calculateTransferBlocks(ctrl, settings);
            fileData = readFile(settings.inputFilename, numBlocks * ctrl.blockSize);
        }

        // Create transfer lists
        TransferMap transfers;
        prepareTransfers(transfers, queues, ctrl, fileData, settings);

        // Do benchmarking
        EventMap writeTimes;
        EventMap readTimes;
        if (settings.write)
        {
            // Write to disk
            benchmark(writeTimes, transfers, settings, true);

            // Clear buffers before reading
            for (const auto& t: transfers)
            {
                t.second->buffer->clear();
            }
        }

        // Read from disk
        benchmark(readTimes, transfers, settings, false);

        // Write data to output file
        if (!settings.outputFilename.empty())
        {
            writeFile(transfers, ctrl, settings);
        }

        // Verify transfers
        if (settings.verify)
        {
            verifyTransfers(fileData, transfers, ctrl, settings);
        }

        // Show some statistics
        if (settings.stats)
        {
            printStatistics(ctrl, readTimes, writeTimes, transfers, settings);
        }

        calculatePercentiles(ctrl, writeTimes, true);
        calculatePercentiles(ctrl, readTimes, false);
    }
    catch (const logic_error& e)
    {
        fprintf(stderr, "\nLogic error: %s\n", e.what());
        return 2;
    }
    catch (const runtime_error& e)
    {
        fprintf(stderr, "FAIL\nUnexpected runtime error: %s\n", e.what());
        return 1;
    }

    fprintf(stderr, "OK!\n");

#ifdef __DIS_CLUSTER__
    SCITerminate();
#endif
    return 0;
}
