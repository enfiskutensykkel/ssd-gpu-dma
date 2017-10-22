#ifndef __DIS_LATENCY_TRANSFER_H__
#define __DIS_LATENCY_TRANSFER_H__

#include <cstddef>
#include <cstdint>
#include <nvm_types.h>
#include "segment.h"



void setDataPointer(nvm_cmd_t* cmd, DmaPtr target, DmaPtr prpList, size_t blockSize, size_t chunkSize);


#endif
