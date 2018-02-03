#ifndef __BUFFER_H__
#define __BUFFER_H__

#include <memory>
#include <cstddef>
#include <cstdint>
#include <nvm_types.h>


typedef std::shared_ptr<nvm_dma_t> BufferPtr;


BufferPtr createBuffer(const nvm_ctrl_t* ctrl, uint32_t adapter, uint32_t id, size_t size);


BufferPtr createBuffer(const nvm_ctrl_t* ctrl, uint32_t adapter, uint32_t id, size_t size, int cudaDevice);


#ifdef __DIS_CLUSTER__
BufferPtr createRemoteBuffer(const nvm_ctrl_t* ctrl, uint32_t adapter, uint32_t number, size_t size);
#endif


#endif
