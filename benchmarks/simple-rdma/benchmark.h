#ifndef __SIMPLE_RDMA_BENCHMARK_H__
#define __SIMPLE_RDMA_BENCHMARK_H__

#include <cstdint>
#include "dma.h"
#include "transfer.h"
#include "queue.h"


uint64_t benchmark(QueueList& queues, const TransferList& transfers, DmaPtr hostBuffer, void* deviceBuffer);

uint64_t benchmark(QueueList& queues, const TransferList& transfers);

#endif
