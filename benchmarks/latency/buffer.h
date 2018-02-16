#ifndef __BENCHMARKS_BUFFER_H__
#define __BENCHMARKS_BUFFER_H__

#include <memory>
#include <cstddef>
#include <cstdint>
#include <nvm_types.h>


typedef std::shared_ptr<nvm_dma_t> DmaPtr;

typedef std::shared_ptr<void> MemPtr;



DmaPtr createHostDma(const nvm_ctrl_t* ctrl, size_t size, uint32_t adapter, uint32_t id);



DmaPtr createHostDma(const nvm_ctrl_t* ctrl, size_t size);



DmaPtr createDeviceDma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice, uint32_t adapter, uint32_t id);



DmaPtr createDeviceDma(const nvm_ctrl_t* ctrl, size_t size, int cudaDevice);



DmaPtr createRemoteDma(const nvm_ctrl_t* ctrl, size_t size, uint32_t adapter, uint32_t number);



MemPtr createHostMemory(size_t size, uint32_t flags);



MemPtr createHostMemory(size_t size);



MemPtr createDeviceMemory(size_t size, int cudaDevice);


#endif
