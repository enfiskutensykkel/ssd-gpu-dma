#include <linux/file.h>
#include <linux/fs.h>
#include <linux/nvme.h>
#include <linux/slab.h>
//#include <linux/genhd.h>
#include "file.h"


struct file_info* get_file_info(int fd)
{
    struct file_info* fi;
    struct file* fp;

    fp = fget(fd);
    if (fp == NULL) 
    {
        printk(KERN_ERR "Invalid file descriptor\n");
        return ERR_PTR(-EBADF);
    }

    // TODO: various checks, see nvme_misc.c in nvme-kmod

    fi = kmalloc(sizeof(struct file_info), GFP_KERNEL);
    if (fi == NULL)
    {
        fput(fp);
        return ERR_PTR(-ENOMEM);
    }

    fi->fd = fd;
    fi->file = fp;
    fi->super_blk = fp->f_inode->i_sb;
    fi->blk_dev = fi->super_blk->s_bdev;

    return fi;
}


void put_file_info(struct file_info* fi)
{
    fput(fi->file);
    kfree(fi);
}
