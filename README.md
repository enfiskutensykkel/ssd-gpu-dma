libnvm: A library for building NVM Express drivers and storage applications
==============================================================================
This library is intended to allow userspace programs to control and manage 
NVM Express (NVMe) disk controllers through an easy-to-use API. The motivation 
is to provide a userspace library your CUDA applications and other programs 
can easily link against, in order to build custom drivers and high-performance 
storage applications.

The library provides simple semantics and functions for mapping userspace
buffers and device memory, providing a simple yet low-level mechanism suitable
for controlling an NVMe disk. By mapping buffers, an NVMe drive is able to 
access these buffers directly (DMA), greatly increasing the IO performance
compared to accessing the drive through normal filesystem abstractions 
provided by the Linux kernel. This eliminates the need to context switch to
kernel space and enables zero-copy access from userspace, greatly reducing 
latency. 

Programs intended for running on GPUs or other computing accelerators that 
support Remote DMA (RDMA), can use this library to enable direct disk access
from the accelerators. Currently, the library supports setting up mappings
for GPUDirect-capable Nvidia GPUs. 
The library also (optionally) uses the 
[SISCI SmartIO API](http://dolphinics.com/products/pcie_smart_io_device_lending.html)
from Dolphin Interconnect Solutions, which allows the programmer to set up
arbitrary configurations of devices and NVMe disks _anywhere_ in a PCIe
cluster, achieving minimal latency in the I/O path.

This README document is organised as follows:
  * The [Quick start](#quick-start) section provides 

Quick start
------------------------------------------------------------------------------
You need a PCIe-attached NVMe disk (rev 1.0 or newer). If the disk contains 
any data, it is important that you back this up before proceeding.

Make sure that the following is installed on your system:
  * CMake 3.1 or newer.
  * GCC version 5.4.0 or newer (compiler must support GNU99 for library and 
    examples, and C++11 for benchmarks and CUDA programs).
  * CUDA 8.0 or newer (optional)
  * Dolphin SISCI API 5.5.0 or newer (optional)

If you plan on using CUDA, you must use the kernel module or Dolphin SmartIO.
See [Kernel module](#using-kernel-module) or [Dolphin SmartIO](#dolphin-smartio).

### How to build ###
Clone repository and enter project root directory.
```
$ mkdir -p build; cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ make libnvme     # builds library
$ make examples    # builds example programs
$ cd module; make  # builds kernel module (optional)
```

### Checking that it works ###
Find out the disk's BDF using `lspci`. For this README, we assume that the
BDF is `05:00.0`.

First unbind the default nvme driver from the disk:
```
$ echo -n "0000:05:00.0" > /sys/bus/pci/devices/0000\:05\:00.0/driver/unbind
```

Then run the identify sample (standing in the build directory). It should
look something like this:
```
$ bin/identify-userspace --ctrl=05:00.0
Resetting controller and setting up admin queues...
------------- Controller information -------------
PCI Vendor ID           : 86 80
PCI Subsystem Vendor ID : 86 80
NVM Express version     : 1.0.0
Controller page size    : 4096
Max queue entries       : 4096
Serial Number           : CVCQ5251008G400AGN
Model Number            : INTEL SSDPEDMW400G4
Firmware revision       : 8EV101H0
Max data transfer size  : 131072
Max outstanding commands: 0
Max number of namespaces: 1
--------------------------------------------------
```


### Using kernel module ###
Using the kernel module is not required, unless you plan on using the 
library with CUDA support.
Currently the only version of Linux supported is Linux 4.11.0. Older version
may work, but you probably have to change the call to `get_user_pages()`
and the DMA API.

You should make sure that you use a processor that supports PCIe peer-to-peer,
for example Intel Xeon, and that you have a GPU with GPUDirect support (Quadro
or Tesla workstation GPUs). For best support, check that your GPU is Pascal 
architecture or newer. Currently, IOMMU support is broken, so disable the
IOMMU. This is done by removing `iommu=on` and `intel_iommu=on` from the 
`CMDLINE` variable in `/etc/default/grub`. 

Loading and unloading the driver is done as follows:
```
$ cd build/module
$ make
$ make load     # will insert the kernel module
$ make unload   # unloads the kernel module
```

You want to unload the default nvme driver for the NVMe disk, and bind 
the disnvm driver to it (replace `05:00.0` with disk BDF):
```
$ echo -n "0000:05:00.0" > /sys/bus/pci/devices/0000\:05\:00.0/driver/unbind
$ echo -n "0000:05:00.0" > /sys/bus/pci/drivers/disnvm/bind
```

After doing this, the file `/dev/disnvm0` should show up, representing the
disk's BAR0.


### Dolphin SmartIO ###
If you have an NTB adapter from Dolphin Interconnect Solutions and are 
familiar with their [SISCI API](http://ww.dolphinics.no/download/ci/docs-master/),
it is possible to use _libnvm_ in conjunction with the SmartIO extension to SISCI.
This provides the user with a flexible method of concurrently sharing one or more
NVMe drives in a PCIe cluster. When using SmartIO, IOMMU is supported but may 
affect PCIe peer-to-peer transfers.

If you have both CUDA and at least SISCI 5.5.0 with SmartIO installed, you can
verify that this works by running the latency benchmark test. You can even run
it on a separate host. Assuming the disk resides on node 4 and is connected to
node 8, do the following on node 4:
```
$ smartio_tool add 05:00.0        # adds the NVMe drive to SmartIO resource pool
$ smartio_tool connect 4          # if you want to run it on the local node as well
$ smartio_tool connect 8          # allow node 8 to borrow devices
$ smartio_tool available 05:00.0  # indicates that the device is available
```

Then do the following on node 8.
```
$ smartio_tool list		  # should show the NVMe disk
80000: Non-Volatile memory controller Intel Corporation Device f1a5 [available]
$ mkdir -p build; cd build
$ make libnvm
$ make latency-benchmark
```

Now run the latency benchmark with the specified controller and for 1000 blocks:
```
$ ./bin/nvm-latency-bench --ctrl=0x80000 --blocks=1000 --pattern=sequential
Resetting controller...
Queue #01 remote qd=32 blocks=1000 offset=0 pattern=sequential (4 commands)
Creating buffer (125 pages)...
Running benchmark...
Queue #01 total-blocks=1000000 count=1000 min=531.366 avg=534.049 max=541.388
	0.99:        540.287
	0.97:        539.424
	0.95:        538.568
	0.90:        535.031
	0.75:        534.377
	0.50:        534.046
	0.25:        533.030
	0.05:        532.025
	0.01:        531.859
OK!
```

You can also compare this with the performance of the disk locally:
```
$ ./bin/nvm-latency-bench --ctrl=0x80000 --blocks=1000 --pattern=sequential
Resetting controller...
Queue #01 remote qd=32 blocks=1000 offset=0 pattern=sequential (4 commands)
Creating buffer (125 pages)...
Running benchmark...
Queue #01 total-blocks=1000000 count=1000 min=536.117 avg=541.190 max=549.240
	0.99:        543.080
	0.97:        542.053
	0.95:        541.825
	0.90:        541.677
	0.75:        541.507
	0.50:        541.346
	0.25:        541.152
	0.05:        539.600
	0.01:        539.351
OK!
```

Note that in this configuration, reads actually have lower latency for the 
remote run than for the local run.


Technical Overview
------------------------------------------------------------------------------

### Non-Volatile Memory Express ###


#### Queue pairs and doorbells

#### Physical Region Pages

### Nvidia GPUDirect ###

### PCIe NTBs and Dolphin SmartIO ###


Library
------------------------------------------------------------------------------
`libnvm` is a userspace library implemented in C for writing custom storage 
applications and/or custom NVMe drivers. By exploiting the memory addressing 
scheme in NVMe, `libnvm` is able to provide a block-level interface with 
extremely low latency in the I/O path. In addition, with minimal driver 
support, it is possible to set up arbitrary memory mappings to device memory, 
enabling peer-to-peer I/O between devices and storage in architectures that 
support it.

The library is in essence similar to [SPDK](http://www.spdk.io/), in that it
moves driver code to user-space and relies on hardware polling rather than 
being interrupt driven. This means that the application can avoid using 
syscalls and has zero-copy data access, as well as getting predictable and
extremely low latency for I/O operations.
As NVMe is designed in a way that reflects the inherent parallelism in modern
CPU architectures, we are able to provide a lock-less interface to the disk
which can be shared by multiple process instances, even running on _separate_
machines(!).

### Header file description ###


  * `nvm_types.h` contains type definitions for the most commonly
    used types. The most interesting types are:
    	
	- `nvm_ctrl_t`: This is the controller reference type.
	  Holds basic information about a controller and a memory
	  map of its doorbell registers.

	- `nvm_dma_t`: DMA descriptor. This is a convenience type
	  for describing memory regions that are mapped for a
          controller.

	- `nvm_queue_t`: Queue descriptor. Used to keep state about
	  I/O queues. Note that the same type is used to represent
	  submission queues (SQs) and completion queues (CQs).

	- `nvm_cmd_t`: Definition of an NVM IO command (SQ entry).

	- `nvm_cpl_t`: Definition of an NVM IO completion (CQ entry).

	- `nvm_aq_ref`: This is a reference to the controller's admin 
	  queue-pair. Used for RPC-like calls to the process that "owns"
	  the admin queue-pair.

  * `nvm_ctrl.h` contains functions for creating and releasing
    a controller reference. It also contains functions for resetting a
    controller.

  * `nvm_dma.h` has helper functions for creating DMA buffer descriptors
    aligned to controller pages. It also has functions for creating mappings
    to memory for the controller.
 
  * `nvm_aq.h` contains the necessary functions for setting up an admin
    queue-pair and creating a reference to this.
    
  * `nvm_rpc.h` contains functions for binding an admin queue-pair reference
    to the actual (remote) admin queue-pair.

  * `nvm_queue.h` consists of "header-only" functions for enqueuing and
    submitting I/O commands as well as polling for completions.

  * `nvm_cmd.h` contains helper functions for building NVM IO commands.

  * `nvm_admin.h` consists of a series of convenience functions for common
    admin commands, such as reserving IO queues and retrieving controller
    and namespace information.

  * `nvm_util.h` is a bunch of convenience macros.

  * `nvm_error.h` deals with packing and unpacking error information.
    Also contains a function similar to `strerror()` to retrieve 
    a human readable error description.


### Typical mode of operation ###

Please refer to section 7 of the NVM Express specification.

### Example programs ###

### Benchmarks and workloads ###

