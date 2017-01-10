#ifndef __SSD_DMA_FILE_H__
#define __SSD_DMA_FILE_H__

#include <linux/fs.h>
#include <linux/file.h>
#include <linux/nvme.h>


struct file_info
{
    int                     fd;
    struct file*            file;
    struct super_block*     super_blk;
    struct block_device*    blk_dev;
};


struct file_info* get_file_info(int user_fd);


void put_file_info(struct file_info* file_info);


#endif
