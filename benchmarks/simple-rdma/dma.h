#ifndef __SIMPLE_RDMA_DMA_H__
#define __SIMPLE_RDMA_DMA_H__

#include <nvm_types.h>
#include <memory>
#include <cstddef>

#ifndef __CUDACC__
#define __host__
#endif


typedef std::shared_ptr<nvm_dma_t> DmaPtr;


__host__ DmaPtr createHostBuffer(nvm_ctrl_t controller, size_t bufferSize);


__host__ DmaPtr createDeviceBuffer(nvm_ctrl_t controller, size_t bufferSize, int cudaDevice);


#endif
