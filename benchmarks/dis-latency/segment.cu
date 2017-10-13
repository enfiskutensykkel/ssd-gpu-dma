#include <cuda.h>
#include <nvm_util.h>
#include <nvm_types.h>
#include <nvm_dma.h>
#include <memory>
#include <string>
#include <stdexcept>
#include <list>
#include <new>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <unistd.h>
#include <sisci_types.h>
#include <sisci_api.h>
#include <sisci_error.h>
#include "segment.h"

using error = std::runtime_error;
using std::string;

struct Segment;
struct Export;
typedef std::shared_ptr<Export> ExportPtr;


struct Export
{
    uint32_t                adapter;
    sci_local_segment_t     segment;

    Export(const Segment& segment, uint32_t adapter);
    ~Export();
};


struct Segment
{
    uint32_t                id;
    size_t                  size;
    sci_desc_t              sd;
    sci_local_segment_t     segment;
    std::list<ExportPtr>    exports;

    Segment(uint32_t id, size_t size, uint32_t flags = 0);

    virtual ~Segment();

    virtual DmaPtr createMapping(nvm_ctrl_t ctrl, uint32_t adapter);

    void disExport(uint32_t adapter);
};


struct CudaSegment: Segment
{
    void*                   memory;
    
    CudaSegment(uint32_t id, size_t size, int cudaDevice);

    virtual ~CudaSegment();

    virtual DmaPtr createMapping(nvm_ctrl_t ctrl, uint32_t adapter);
};


Export::Export(const Segment& segment, uint32_t adapter)
    : adapter(adapter)
    , segment(segment.segment)
{
    sci_error_t err;

    SCIPrepareSegment(this->segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        throw error("Failed to prepare segment: " + string(SCIGetErrorString(err)));
    }

    SCISetSegmentAvailable(this->segment, adapter, 0, &err);
    if (err != SCI_ERR_OK)
    {
        throw error("Failed to set segment available: " + string(SCIGetErrorString(err)));
    }
}


Export::~Export()
{
    sci_error_t err;

    do
    {
        SCISetSegmentUnavailable(segment, adapter, 0, &err);
    }
    while (err == SCI_ERR_BUSY);
}


void Segment::disExport(uint32_t adapter)
{
    for (const auto& segmentExport: exports)
    {
        if (segmentExport->adapter == adapter)
        {
            return;
        }
    }

    exports.push_back(std::make_shared<Export>(*this, adapter));
}


Segment::Segment(uint32_t id, size_t size, uint32_t flags)
    : id(id)
    , size(size)
    , sd(nullptr)
    , segment(nullptr)
{
    sci_error_t err;

    SCIOpen(&sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        throw error("Failed to open SISCI virtual descriptor:" + string(SCIGetErrorString(err)));
    }

    SCICreateSegment(sd, &segment, id, size, nullptr, nullptr, flags, &err);
    if (err != 0)
    {
        string errorString("Failed to create segment: " + string(SCIGetErrorString(err)));
        SCIClose(sd, 0, &err);
        throw error(errorString);
    }
}


Segment::~Segment()
{
    sci_error_t err;

    exports.clear();

    do
    {
        SCIRemoveSegment(segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

    SCIClose(sd, 0, &err);
}


CudaSegment::CudaSegment(uint32_t id, size_t size, int device)
    : Segment(id, size, SCI_FLAG_EMPTY)
    , memory(nullptr)
{
    cudaError_t err;
    sci_error_t scierr;

    err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        throw error("Failed to set CUDA device: " + string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&memory, size);
    if (err != cudaSuccess)
    {
        throw error("Failed to allocate memory: " + string(cudaGetErrorString(err)));
    }

    cudaPointerAttributes attrs;
    err = cudaPointerGetAttributes(&attrs, memory);
    if (err != cudaSuccess)
    {
        cudaFree(memory);

    }

    SCIAttachPhysicalMemory(0, attrs.devicePointer, 0, size, segment, SCI_FLAG_CUDA_BUFFER, &scierr);
    if (err != 0)
    {
        cudaFree(memory);
        throw error("Failed to attach physical memory: " + string(SCIGetErrorString(scierr)));
    }
}


CudaSegment::~CudaSegment()
{
    cudaFree(memory);
}


SegmentPtr createSegment(uint32_t id, size_t size, int cudaDevice)
{
    if (cudaDevice < 0)
    {
        long pageSize = sysconf(_SC_PAGESIZE);
        if (pageSize == -1)
        {
            throw error("Failed to get page size: " + string(std::strerror(errno)));
        }

        return std::make_shared<Segment>(id, DMA_SIZE(size, pageSize));
    }

    const size_t boundSize = 0x10000;
    return std::make_shared<CudaSegment>(id, DMA_SIZE(size, boundSize), cudaDevice);
}


static void deleteHandle(nvm_dma_t* handle)
{
    nvm_dma_window_free(*handle);
    delete handle;
}


DmaPtr Segment::createMapping(nvm_ctrl_t ctrl, uint32_t adapter)
{
    sci_error_t err;
    sci_map_t map;

    nvm_dma_t* handle = new (std::nothrow) nvm_dma_t;
    if (handle == nullptr)
    {
        throw error("Failed to allocate DMA mapping handle");
    }

    try
    {
        disExport(adapter);
    }
    catch (const error& err)
    {
        delete handle;
        throw err;
    }
    
    void* ptr = SCIMapLocalSegment(segment, &map, 0, size, nullptr, 0, &err);
    if (err != SCI_ERR_OK)
    {
        delete handle;
        throw error("Failed to map segment memory to virtual address space: " + string(SCIGetErrorString(err)));
    }

    int nvmerr = nvm_dis_dma_window_init(handle, ctrl, adapter, segment, ptr, size);
    if (nvmerr != 0)
    {
        do
        {
            SCIUnmapSegment(map, 0, &err);
        }
        while (err == SCI_ERR_BUSY);

        delete handle;
        throw error("Failed to create DMA mapping: " + string(std::strerror(nvmerr)));
    }

    return DmaPtr(handle, [map](nvm_dma_t* handle) {
        sci_error_t err;

        deleteHandle(handle);

        do
        {
            SCIUnmapSegment(map, 0, &err);
        }
        while (err == SCI_ERR_BUSY);
    });
}


DmaPtr CudaSegment::createMapping(nvm_ctrl_t ctrl, uint32_t adapter)
{
    nvm_dma_t* handle = new (std::nothrow) nvm_dma_t;
    if (handle == nullptr)
    {
        throw error("Failed to allocate DMA mapping handle");
    }

    try
    {
        disExport(adapter);
    }
    catch (const error& err)
    {
        delete handle;
        throw err;
    }

    int nvmerr = nvm_dis_dma_window_init(handle, ctrl, adapter, segment, memory, size);
    if (nvmerr != 0)
    {
        delete handle;
        throw error("Failed to create DMA mapping: " + string(std::strerror(nvmerr)));
    }

    return DmaPtr(handle, deleteHandle);
}


void exportSegment(SegmentPtr segment, uint32_t adapter)
{
    segment->disExport(adapter);
}


DmaPtr createDmaMapping(SegmentPtr segment, nvm_ctrl_t ctrl, uint32_t adapter)
{

    return segment->createMapping(ctrl, adapter);
}

