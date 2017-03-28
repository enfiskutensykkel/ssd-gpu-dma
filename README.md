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

 * Linux kernel (preferrably version 
   4.4.21, 4.4.0.57, 4.4.0.66, 4.4.0.67, or 4.4.0.70, 
   since they're the ones I've used). You might need to change the call to 
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


Limitations
---------------------------------
Due to convenience, there are some limitiations that must be mentioned:

 * I make the assumption that the size of a page of memory is 4 KiB 
   (4096 bytes). This goes for system memory, GPU memory and page regions
   used by the NVM controller.
   
 * Although the [NVM specification](#nvm-express) specifies the possibility
   for a controller to support queues using non-contiguous buffers (fragmented
   over several  pages), I only use page-sized buffers throughout my code. 
   This is also done out of convenience, as the support for non-contiguous 
   buffers are merely _optional_ in NVM  Express.

 * At the moment, this project is limited to x86\_64 and I currently do __not__
   support using an 
   [IOMMU](https://en.wikipedia.org/wiki/Input%E2%80%93output_memory_management_unit).
   In other words, the project assumes that physical addresses are directly 
   translatable into valid IO addresses.



Program Overview
==================================
The program is divided into three main parts,

 1. a skeleton kernel module responsible for page-locking user-space memory
    and returning a valid IO address

 2. a minimalistic NVM driver implemented in user-space by using _sysfs_ and
    the before-mentioned kernel module, responsible for setting up the disk
    and creating IO submission/completion queues

 3. CUDA code accessing the disk using the submission and completion queues 
    created by the driver


Skeleton Kernel Module
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
assumes that there is no IOMMU enabled. Please ensure that you have turned 
off _VT-d_ in your BIOS settings and `cat /proc/cmdline | grep iommu` shows 
nothing (you can also check `dmeg | grep DMAR`).


User-space Program
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
---------------------------------
[NVM Express](http://www.nvmexpress.org/wp-content/uploads/NVM_Express_1_2_1_Gold_20160603.pdf)
is a standard specifying how disk controllers should behave and interact with
a host system. The main benefit of NVM Express disks comes from the fact that
NVM commands are hosted on system memory rather than on controller memory.

In NVM Express, a disk controller reads commands from an IO submission queue,
performs the operation, and posts a completion to an IO completion queue (and
triggering an optional MSI-X interrupt, if required). A controller must 
support a bunch of these queue pairs, meaning that different CPUs can have 
their own queue pair and operate on them without needing any form of 
synchronization.

These two features of NVM Express is quite powerful, and allows people like me
to host IO queues in other places than system memory, i.e. on GPU memory. 
Since the NVM controller can now be controlled from multiple processing units
without any form of synchronization, a GPU can now control the disk without
involving the CPU or needing to go through system memory for data transfers.


sysfs
---------------------------------
Linux offers a file system structured interface for accessing PCI devices, 
called [sysfs](https://www.kernel.org/doc/Documentation/filesystems/sysfs-pci.txt). 
PCI devices have designated memory regions, such as _config space_ and 
different _Base Address Registers_, that drivers use to access the device.



Nvidia CUDA
---------------------------------
Nvidia CUDA is Nvidia's framework and development kit for creating general
purpose software that runs on their GPUs. Nvidias workstation GPUs, namely the
Quadro and Tesla models, support something Nvidia calls _GPUDirect_ which 
offers additional DMA capabilities.

 * [GPUDirect RDMA](http://docs.nvidia.com/cuda/gpudirect-rdma/#axzz4a6tqNDAe)

 * [GPUDirect Async](http://on-demand.gputechconf.com/gtc/2016/presentation/s6264-davide-rossetti-GPUDirect.pdf)



Possible Optimizations
---------------------------------
NVM Express specifies that a controller may optionally have on-board general
purpose memory, where it can host (among other things) IO submission queues.
Since the controller needs to first do DMA to fetch the IO command (because
reads are _posted transactions_ in PCI Express), it could be a performance
optimization to host the queue on the controller itself.

In our case, we could use GPUDirect Async to set up IO commands on the
controller rather than in GPU memory, and only host the completion queue
on the GPU, ensuring that we push data rather than pulling it.


Howto
==================================
**Note:** This section is incomplete as I haven't finished my program yet.

 1. Verify that the IOMMU is disabled.
   
   * check `cat /proc/cmdline`
   * remove any `iommu=on intel_iommu=on` in GRUB config
   * check the result of running `dmesg | grep DMAR`
   * disable IOMMU in BIOS (Intel VT-d, AMD-Vi)

 2. Compile the kernel module and program

   * run `make` from the project root directory

 3. Find a suitable SSD
   
   * search the output from `lspci -tv` for a suitable device
   * note the device BFD

 4. Make a new nvme namespace

 5. Find a suitable GPU

   * the tool `nvidia-smi` lists Nvidia GPUs

 6. Unbind the SSD from the `nvme` driver
 
   * run the script `./unbind.sh unbind <BDF>` as user root

 7. Load the kernel module

   * run `make load` as root
   * check `dmesg` and look for the line `cunvme loaded`
   * check that the file `/proc/cunvme` exists

 8. Run the program

   * run `./cuda-nvme --controller=<BDF> --identify` as user root

