#ifndef __BENCHMARK_BUFFER_H__
#define __BENCHMARK_BUFFER_H__

#include <nvm_types.h>
#include <memory>
#include <cstddef>
#include <cstdint>



typedef std::shared_ptr<nvm_dma_t> DmaPtr;

typedef std::shared_ptr<void> BufferPtr;



DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size);


DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice);


#ifdef __DIS_CLUSTER__
DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, uint32_t adapter, uint32_t id);
#endif


#ifdef __DIS_CLUSTER__
DmaPtr createDma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice, uint32_t adapter, uint32_t id);
#endif


BufferPtr createBuffer(size_t size);


BufferPtr createBuffer(size_t size, int cudaDevice);


#ifdef __DIS_CLUSTER__
DmaPtr createRemoteDma(const nvm_ctrl_t* ctrl, size_t size, uint32_t adapter, uint32_t segno);
#endif

#endif
