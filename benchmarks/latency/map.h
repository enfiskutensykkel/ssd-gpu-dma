#ifndef __BENCHMARKS_MAP_H__
#define __BENCHMARKS_MAP_H__

#include <cstddef>
#include <cstdint>
#include <memory>
#include "buffer.h"

struct Segment;

typedef std::shared_ptr<Segment> SegPtr;


SegPtr createGpuMapping(MemPtr gpuBuffer, size_t size, uint64_t gpuDeviceId, uint32_t gpuAdapter, uint32_t segmentId);


DmaPtr createDeviceDma(const nvm_ctrl_t* ctrl, uint32_t adapter, SegPtr ptr);

#endif
