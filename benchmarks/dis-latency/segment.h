#ifndef __DIS_LATENCY_SEGMENT_H__
#define __DIS_LATENCY_SEGMENT_H__

#include <memory>
#include <cstddef>
#include <cstdint>
#include <sisci_types.h>
#include <nvm_types.h>


struct Segment;
typedef std::shared_ptr<Segment> SegmentPtr;

typedef std::shared_ptr<nvm_dma_t> DmaPtr;


SegmentPtr createSegment(uint32_t id, size_t size, int cudaDevice = -1);


void exportSegment(SegmentPtr segment, uint32_t adapter);


DmaPtr createDmaMapping(SegmentPtr segment, nvm_ctrl_t ctrl, uint32_t adapter);


#endif
