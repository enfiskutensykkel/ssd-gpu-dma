cunvme = Nvidia CUDA + NVM Express
==================================

This project demonstrates how a NVM Express device can be accessed and 
controlled from a CUDA program. This program imeplements a barebones NVM driver 
in user-space and passes the control to a GPU thread (CUDA kernel), which is 
then able to use the disk as a generic block storage device.

The motivation for doing this is to show that disk IO can be performed
directly between a PCIe SSD and a GPU without involving the CPU or system
memory (RAM) for transferring data.



Requirements
---------------------------------
 * x86\_64 architecture with page size set to 4096 bytes.

 * Linux kernel (preferrably version 4.4.21 or 4.4.57, since they're 
   the ones I've used). You might need to change the call to 
   `get_user_pages()` if you use a different kernel version.

 * A NVM Express compatible disk. 
   The disk I've tested with is an Intel Corporation SSD 750 Series.

 * CUDA 8.0 SDK or newer and associated drivers, which are needed for 
   [GPUDirect Async](#nvidia-cuda) support (Nvidia's fancy market name doing 
   DMA from the GPU to third-party devices). Note that the location of the 
   `nvidia/nv-p2p.h` header might change between different driver versions.
    
 * A Nvidia Tesla or Quadro model GPU, with Maxwell or Pascal (or newer).
   I've tested with a Quadro K620. Only Quadros or Teslas are able to do
   [GPUDirect RDMA](#nvidia-cuda) and only Maxwell or newer are able to do 
   GPUDirect Async.



Software architecture overview
---------------------------------
The program is divided into three main parts,

 1. a skeleton kernel module responsible for page-locking user-space memory
    and returning a valid IO address

 2. a minimalistic NVM driver implemented in user-space by using _sysfs_ and
    the before-mentioned kernel module, responsible for setting up the disk
    and creating IO submission/completion queues

 3. CUDA code accessing the disk using the submission and completion queues 
    created by the driver


Kernel module
---------------------------------
The skeleton kernel module is deliberately minimalistic, and is only 
respoinsible for page-locking memory pages either in RAM or on the GPU, as
Nvidia GPUs implement their own memory model with virtual address space.

When the kernel module loads, it creates a file, `/proc/cunvme`, which a user-
space program uses to send `ioctl` commands. The kernel module performs the
requested commands, such as pinning a page-sized memory buffer in memory, and 
returns a valid bus/IO address for the page.

Supported operations performed by the kernel module are:
 * pin a page of RAM memory and returning the IO address
 * unpin a page of previously pinned RAM memory
 * pin a page of GPU memory and returning an IO address 
 * unpin a page of previously pinned GPU memory

For simplicity, the kernel module only operates on single memory pages and 
assumes that there is no [IOMMU](https://en.wikipedia.org/wiki/Input%E2%80%93output_memory_management_unit)
enabled. Please ensure that you have turned off _VT-d_ in your BIOS settings
and `cat /proc/cmdline | grep iommu` shows nothing (you can also check `dmeg | grep DMAR`).



User-space program
---------------------------------
The user-space application is implemented partially in C and CUDA-style C++.
The user specifies the __full__ PCI BDF (`xxxx:yy:zz.0`) for the SSD disk and,
optionally, which CUDA device to use (defaults to device 0). The program then
accesses the disk by memory mapping its [sysfs](#sysfs) resources and sets up
the necessary NVM submission and completion queues hosted in GPU memory.

After this, the program starts up a couple of GPU threads (CUDA kernels) and
allow them to perform persistent reads and writes to specific disk blocks on
the specified NVM namespace.



NVM Express
--------------------------------------


[NVM Express](http://www.nvmexpress.org/wp-content/uploads/NVM_Express_1_2_1_Gold_20160603.pdf)

For simplicity is every queue a page large

sysfs
--------------------------------------
[Using sysfs to access PCI device resources](https://www.kernel.org/doc/Documentation/filesystems/sysfs-pci.txt)

Nvidia CUDA
--------------------------------------
  - [GPUDirect RDMA](http://docs.nvidia.com/cuda/gpudirect-rdma/#axzz4a6tqNDAe)

  - [GPUDirect Async](http://on-demand.gputechconf.com/gtc/2016/presentation/s6264-davide-rossetti-GPUDirect.pdf)

Loadable Kernel Module
--------------------------------------
Linux 4.4.0 family due to changing signature of `get_user_pages`

Limitations
--------------------------------------
IOMMU stuff
pages allocated in userspace are not contiguous (why everything is operating on pages)


Future optimisations
--------------------------------------
  - Use controller onboard memory (optional in NVMe) to host queues and push data using GPUDirect Async
