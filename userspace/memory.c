#include "memory.h"
#include <sisci_types.h>
#include <sisci_api.h>
#include <unistd.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>


#define GPU_PAGE_SIZE   (1L << 16) // GPU bound pointers needs to be aligned to 64 KB


static sci_error_t create_dma_window(buffer_t* handle)
{
    size_t i;
    sci_error_t err;
    sci_ioaddr_t addr;

    SCIMapSegmentForDevice(handle->segment, handle->bus_handle, &addr, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to map for device: %x\n", err);
        return err;
    }

    // Calculate logical bus addresses
    for (i = 0; i < handle->n_addrs; ++i)
    {
        size_t phys_page_idx = (i * handle->unit_size) / handle->page_size;
        size_t offset_within_page = (i * handle->unit_size) % handle->page_size;

        uint64_t logical_bus_addr = addr + (phys_page_idx * handle->page_size) + offset_within_page;
        handle->bus_addr[i] = logical_bus_addr;
    }

    handle->n_addrs = i;

    return err;
}


static long get_page_size(int dev)
{
    long page_size = GPU_PAGE_SIZE;

    if (dev < 0)
    {
        page_size = sysconf(_SC_PAGESIZE);

        if (page_size == -1)
        {
            fprintf(stderr, "Failed to get system page size: %s\n", strerror(errno));
        }
    }

    return page_size;
}


buffer_t* get_buffer(int dev, int id, size_t buffer_size, size_t nvm_page_size, uint64_t device_id)
{
    sci_error_t err;

    long page_size = get_page_size(dev);
    if (page_size == -1)
    {
        return NULL;
    }

    size_t range_size = (buffer_size + page_size - 1) & ~(page_size - 1);
    size_t nvm_pages = range_size / nvm_page_size;

    buffer_t* handle = (buffer_t*) malloc(sizeof(buffer_t) + sizeof(uint64_t) * nvm_pages);
    if (handle == NULL)
    {
        fprintf(stderr, "Failed to allocate memory handle: %s\n", strerror(errno));
        goto exit;
    }

    SCIOpen(&handle->sd, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to open SISCI descriptor\n");
        goto exit;
    }

    SCICreateSegment(handle->sd, &handle->segment, id, buffer_size, NULL, NULL, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to create segment\n");
        goto close;
    }

    handle->id = id;
    handle->device = dev;
    handle->virt_addr = NULL;
    handle->range_size = range_size;
    handle->page_size = page_size;
    handle->unit_size = nvm_page_size;
    handle->bus_handle = device_id;
    handle->n_addrs = nvm_pages;

    if (handle->device >= 0)
    {
        fprintf(stderr, "CUDA not supported yet\n");
        goto remove;
    }
    else
    {
    }

    SCIPrepareSegment(handle->segment, 0, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to prepare segment\n");
        goto remove;
    }

    SCISetSegmentAvailable(handle->segment, 0, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to set segment available\n");
        goto remove;
    }

    err = create_dma_window(handle); 
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to create DMA window\n");
        goto remove;
    }

    handle->virt_addr = SCIMapLocalSegment(handle->segment, &handle->mapping, 0, handle->range_size, NULL, 0, &err);
    if (err != SCI_ERR_OK)
    {
        fprintf(stderr, "Failed to memory map segment\n");
        goto remove;
    }

    return handle;

remove:
    do
    {
        SCIRemoveSegment(handle->segment, 0, &err);
    }
    while (err == SCI_ERR_BUSY);

close:
    SCIClose(handle->sd, 0, &err);

exit:
    free(handle);
    return NULL;
}


int get_page(int dev, int id, page_t* page, uint64_t bus_handle)
{
    long page_size = get_page_size(dev);
    if (page_size == -1)
    {
        return EIO;
    }

    buffer_t* handle = get_buffer(dev, id, page_size, page_size, bus_handle);
    if (handle == NULL)
    {
        fprintf(stderr, "Failed to allocate and pin page\n");
        return ENOMEM;
    }

    page->sd = handle->sd;
    page->segment = handle->segment;
    page->mapping = handle->mapping;
    page->id = handle->id;
    page->device = handle->device;
    page->virt_addr = handle->virt_addr;
    page->page_size = handle->page_size;
    page->bus_handle = handle->bus_handle;
    page->bus_addr = handle->bus_addr[0];

    free(handle);
    return 0;
}


void put_buffer(buffer_t* handle)
{
    sci_error_t err;

    if (handle != NULL)
    {
        if (handle->device >= 0)
        {
            fprintf(stderr, "CUDA not supported yet\n");
        }
        else
        {
            do
            {
                SCISetSegmentUnavailable(handle->segment, 0, 0, &err);
            }
            while (err == SCI_ERR_BUSY);

            do
            {
                SCIUnmapSegment(handle->mapping, 0, &err);
            }
            while (err == SCI_ERR_BUSY);

            do
            {
                SCIRemoveSegment(handle->segment, 0, &err);
            }
            while (err == SCI_ERR_BUSY);

            SCIClose(handle->sd, 0, &err);
        }

        free(handle);
    }
}


void put_page(page_t* page)
{
    sci_error_t err;

    if (page != NULL)
    {
        if (page->device >= 0)
        {
            fprintf(stderr, "CUDA not supported yet\n");
        }
        else
        {
            do
            {
                SCISetSegmentUnavailable(page->segment, 0, 0, &err);
            }
            while (err == SCI_ERR_BUSY);

            do
            {
                SCIUnmapSegment(page->mapping, 0, &err);
            }
            while (err == SCI_ERR_BUSY);

            do
            {
                SCIRemoveSegment(page->segment, 0, &err);
            }
            while (err == SCI_ERR_BUSY);

            SCIClose(page->sd, 0, &err);
        }
        page->virt_addr = NULL;
    }
}

