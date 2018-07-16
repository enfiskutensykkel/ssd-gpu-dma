#ifndef __LATENCY_BENCHMARK_UTILITY_H__
#define __LATENCY_BENCHMARK_UTILITY_H__

#include <nvm_util.h>
#include <stdexcept>
#include <cstddef>
#include "ctrl.h"
#include "settings.h"



static size_t calculateTransferBlocks(const Ctrl& ctrl, const Settings& settings)
{
    size_t size = 0;

    switch (settings.unit)
    {
        case AccessUnit::BLOCKS:
            size = settings.count;
            break;

        case AccessUnit::PAGES:
            size = NVM_PAGE_ALIGN(settings.count * ctrl.pageSize, ctrl.blockSize) / ctrl.blockSize;
            break;

        case AccessUnit::BYTES:
            size = NVM_PAGE_ALIGN(settings.count, ctrl.blockSize) / ctrl.blockSize;
            break;

        default:
            throw std::logic_error("Unknown transfer unit");
    }

    return size;
}



static size_t calculateOffsetBlocks(const Ctrl& ctrl, const Settings& settings)
{
    size_t offset = 0;

    switch (settings.unit)
    {
        case AccessUnit::BLOCKS:
            offset = settings.offset;
            break;

        case AccessUnit::PAGES:
            offset = NVM_PAGE_TO_BLOCK(ctrl.pageSize, ctrl.blockSize, settings.offset);
            break;

        case AccessUnit::BYTES:
            offset = NVM_PAGE_ALIGN(settings.offset, ctrl.blockSize) / ctrl.blockSize;
            break;

        default:
            throw std::logic_error("Unknown transfer unit");
    }

    return offset;
}



static size_t calculateBufferSize(const Ctrl& ctrl, const Settings& settings)
{
    size_t size = 0;

    switch (settings.unit)
    {
        case AccessUnit::BLOCKS:
            size = NVM_PAGE_ALIGN(settings.count * ctrl.blockSize, ctrl.pageSize);
            break;

        case AccessUnit::PAGES:
            size = settings.count * ctrl.pageSize;
            break;
            
        case AccessUnit::BYTES:
            size = NVM_PAGE_ALIGN(settings.count, ctrl.pageSize);
            break;

        default:
            throw std::logic_error("Unknown transfer unit");
    }

    return size;
}

#endif /* __LATENCY_BENCHMARK_UTILITY_H__ */
