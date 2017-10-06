#ifndef __SIMPLE_RDMA_BENCHMARK_H__
#define __SIMPLE_RDMA_BENCHMARK_H__

#include <cstdint>
#include "dma.h"
#include "transfer.h"
#include "queue.h"


uint64_t currentTime();

uint64_t benchmark(QueueList& queues, const TransferList& transfers);

uint64_t benchmarkWrite(QueueList& queues, const TransferList& transfers);

#endif
