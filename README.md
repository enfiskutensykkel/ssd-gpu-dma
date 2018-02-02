libnvm: An API for building userspace NVMe drivers and storage applications
===============================================================================
This library is intended to allow userspace programs to control and manage 
NVM Express (NVMe) [NVMe1.3] disk controllers through an easy-to-use API. 
The motivation is to provide a userspace library your CUDA applications and 
other programs can easily link against, in order to build custom drivers and 
high-performance storage applications.

The library uses simple semantics and functions for mapping userspace memory
buffers and device memory, providing a simple yet low-level mechanism suitable
for controlling an NVMe disk and performing IO operations. By mapping user-
space memory directly, IO performance is greatly increased compared to 
accessing storage through normal file system abstractions provided by the
Linux kernel.

The API can also be linked with applications using the SISCI SmartIO API from 
Dolphin Interconnect Solutions. This allows the programmer to set up arbitrary
configurations of devices and NVMe disks in a PCIe cluster, and enables 
concurrent low-latency access to one or more NVMe disks from multiple machines 
in the cluster.



Quick start
-------------------------------------------------------------------------------
You need a PCIe-attached or M.2 NVMe disk (not the system disk!). If the disk 
contains any data, you should back this up before proceeding. It is also highly 
recommended that you  read the NVMe specification first, which can be found at 
the following URL: <http://nvmexpress.org/resources/specifications/>


### Prerequisites and requirements ###
Please make sure that the following is installed on your system:
* A relatively new Linux kernel
* CMake 3.1 or newer.
* GCC version 5.4.0 or newer. Compiler must support GNU extensions
  for C99 and linking with POSIX threads is required.

The above is sufficient for building the userspace library and most of the 
example programs.

For linking with CUDA programs, you need the following:
* An Nvidia GPU capable of GPUDirect RDMA and GPUDirect Async [GPUDirect].
  This means either a Quadro or Tesla workstation model using the Kepler 
  architecture or newer [GPUDirect Async].
* An architecture that supports PCIe peer-to-peer, for example the Intel Xeon
  family of processors.
* The _FindCUDA_ package for CMake.
* GCC version 5.4.0 or newer. Compiler must be able to compile C++11.
* CUDA 8.0 or newer with CUDA development toolkit.
* Kernel module symbols and headers for your Nvidia driver.

For linking with SISCI API, you additionally need the Dolphin 5.5.0 software
base (or newer) with CUDA support and SmartIO enabled. If you are _not_ using
Dolphin software, you need to explicitly disable IOMMU as current Nvidia GPUs
do not currently fully support it. This is done by removing `iommu=on` and 
`intel_iommu=on` from the `CMDLINE` variable in `/etc/default/grub` and then
reconfiguring GRUB before rebooting. If you _are_ using SmartIO, the Dolphin 
driver stack will handle this for you and it is recommended that you leave the 
IOMMU on for memory protection.


### Building the project ###
From the project root directory, do the following:
```
$ mkdir -p build; cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release # use =Debug for debug build
$ make libnvm                         # builds library
$ make examples                       # builds example programs
$ cd module; make                     # only required if no SISCI SmartIO
```

If you are going to use CUDA, you also need to locate the kernel module
directory and manually run `make`. Locations will vary on different distros
and based on installation type, but on Ubuntu the driver source can be found 
in `/usr/src/nvidia-<major>-<major>.<minor>` if you install CUDA through the
`.deb.` package.

The CMake configuration is _supposed to_ autodetect the location of CUDA, 
Nvidia driver and SISCI library. CUDA is located by the _FindCUDA_ package for
CMake, while the location of both the Nvidia driver and SISCI can be manually
set by overriding the `NVIDIA` and `DIS` defines for CMake 
(`cmake .. -DNVIDIA=/usr/src/...` -DDIS=/opt/DIS/`).

If you have disabled the IOMMU, you can run the _identify_ example to verify
that your build is working. Find out your disk's PCI BDF by using `lspci`.
In our example, assume that it is `05:00.0`.

First unbind the default nvme driver from the disk:
```
$ echo -n "0000:05:00.0" > /sys/bus/pci/devices/0000\:05\:00.0/driver/unbind
```

Then run the identify sample (standing in the build directory). It should
look something like this:
```
$ make libnvm && make identify
$ ./bin/nvm-identify-userspace --ctrl=05:00.0
Resetting controller and setting up admin queues...
------------- Controller information -------------
PCI Vendor ID           : 86 80
PCI Subsystem Vendor ID : 86 80
NVM Express version     : 1.2.0
Controller page size    : 4096
Max queue entries       : 256
Serial Number           : BTPY74400DQ5256D
Model Number            : INTEL SSDPEKKW256G7
Firmware revision       :  PSF121C
Max data transfer size  : 131072
Max outstanding commands: 0
Max number of namespaces: 1
--------------------------------------------------
```

If you are using SISCI SmartIO, you need to use the SmartIO utility program to
configure the disk for device sharing.
```
$ smartio_tool add 05:00.0
$ smartio_tool available 05:00.0
$ smartio_tool connect <local node id>
$ smartio_tool list
80000: Non-Volatile memory controller Intel Corporation Device f1a5 [available]
$ make libnvm && make identify
$ ./bin/nvm-identify --ctrl=0x80000
Resetting controller and setting up admin queues...
------------- Controller information -------------
PCI Vendor ID           : 86 80
PCI Subsystem Vendor ID : 86 80
NVM Express version     : 1.2.0
Controller page size    : 4096
Max queue entries       : 256
Serial Number           : BTPY74400DQ5256D
Model Number            : INTEL SSDPEKKW256G7
Firmware revision       :  PSF121C
Max data transfer size  : 131072
Max outstanding commands: 0
Max number of namespaces: 1
Current number of CQs   : 8
Current number of SQs   : 8
--------------------------------------------------
```



Non-Volatile Memory Express (NVMe)
-------------------------------------------------------------------------------
NVMe [NVMe1.3] is a software specification for disk controllers (_drives_) that 
provides storage on non-volatile media, for example flash memory or Intel's
3D XPoint [3D XPoint].


The specification is designed in a way that reflects the parallelism in modern
CPU architectures: a controller can support up to 2^16 - 1 IO queues with up
to 64K outstanding commands per queue. It does not require any register reads
in the command or completion path, and it requires a maximum of a 32-bit 
register write in the command submission path to a dedicated register.

The specification assumes an underlying bus interface that conforms to PCIe.

### NVM Namespaces 
A namespace is a quantity of non-volatile memory that may be formatted into
logical blocks. A NVMe controller may support multiple namespaces. 
Many controllers may attach the same namespace. In many ways, a namespace
can be regarded as an abstraction of traditional disk partitions.


### Queue pairs and doorbells 
NVMe is based on a paired submission and completiong queue mechanism.
The software will enqueue commands on the submission queue (SQ), and
completions are posted by the controller to the associated completion
queue (CQ). Multiple SQs may use the same CQ, and queues are allocated
in system memory. In other words, there are an N:M mapping of SQs and CQs.

Typically the number of command queues are based on the number of CPU cores.
For example, on a four core processor, there may be a queue pair per core to
avoid locking and ensure that commands are local to the appropriate 
processors' cache. 

A SQ is a ring buffer with a fixed slot size that software uses to submit
commands for execution by the controller. After the command structure is 
updated in memory, the software updates the appropriate SQ tail doorbell
register with the number of commands to execute. The controller fetches
the SQ entries in order from the SQ, but may execute them in an arbitrary
order. Each entry in the SQ is a command. Commands are 64 bytes in size. 

An admin submission queue (ASQ) and completion queue (ACQ) exists for the 
purpose of controller management and control. There is a dedicated command
set for admin commands.


### Physical Region Pages and Scatter-Gather Lists 




Quick start
------------------------------------------------------------------------------
You need a PCIe-attached NVMe disk (version 1.0 or newer). If the disk 
contains any data, it is important that you back this up before proceeding. 
It is also highly recommended that you read the NVMe specification, which can 
be found at the following link: 
<http://nvmexpress.org/resources/specifications/>


### Requirements and prerequisites
Make sure that the following is installed on your system:
  * Linux kernel version 4.11.0-14 or newer
  * CMake 3.1 or newer.
  * GCC version 5.4.0 or newer (compiler must support GNU99 for library and 
    examples, and C++11 for benchmarks and CUDA programs).
  * CUDA 8.0 or newer
  * Dolphin SISCI API 5.5.0 or newer (optional)

If you plan on using CUDA, you must use the kernel module or Dolphin SmartIO.
See the sections on [using the kernel module](#using-kernel-module) or 
[using Dolphin SmartIO](#using-dolphin-smartio).


### How to build ###
Clone repository and enter project root directory.


### Checking that it works ###


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
IOMMU. 

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


### Using Dolphin SmartIO ###
If you have an NTB adapter from Dolphin Interconnect Solutions and are 
familiar with their [SISCI API]
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



Nvidia GPUDirect
------------------------------------------------------------------------------

Programs intended for running on GPUs or other computing accelerators that 
support Remote DMA (RDMA), can use this library to enable direct disk access
from the accelerators. Currently, the library supports setting up mappings
for GPUDirect-capable Nvidia GPUs. 



PCIe NTBs and Dolphin SmartIO
------------------------------------------------------------------------------





API overview
------------------------------------------------------------------------------
`libnvm` is a userspace library implemented in C for writing custom storage 
applications and/or custom NVMe drivers. By exploiting the memory addressing 
scheme in NVMe, `libnvm` is able to provide a block-level interface with 
extremely low latency in the IO path. In addition, with minimal driver 
support, it is possible to set up arbitrary memory mappings to device memory, 
enabling peer-to-peer IO between devices and storage in architectures that 
support it.

The library is in essence similar to [SPDK], in that it
moves driver code to user-space and relies on hardware polling rather than 
being interrupt driven. This means that the application can avoid using 
syscalls and has zero-copy data access, as well as getting predictable and
extremely low latency for I/O operations.
As NVMe is designed in a way that reflects the inherent parallelism in modern
CPU architectures, we are able to provide a lock-less interface to the disk
which can be shared by multiple process instances, even running on _separate_
machines(!).



 eliminates the need to context switch to
kernel space and enables zero-copy access from userspace, greatly reducing 
latency. 
	
	
, achieving minimal latency in the I/O path.

### Scope and limitations of `libnvm`

### Types

* `nvm_ctrl_t`: This is the controller reference type.
  Holds basic information about a controller and a memory
  map of its doorbell registers.

* `nvm_dma_t`: DMA descriptor. This is a convenience type
  for describing memory regions that are mapped for a
  controller.

* `nvm_queue_t`: Queue descriptor. Used to keep state about
  I/O queues. Note that the same type is used to represent
  submission queues (SQs) and completion queues (CQs).

* `nvm_cmd_t`: Definition of an NVM IO command (SQ entry).

* `nvm_cpl_t`: Definition of an NVM IO completion (CQ entry).

* `nvm_aq_ref`: This is a reference to the controller's admin 
  queue-pair. Used for RPC-like calls to the process that "owns"
  the admin queue-pair.


### Header files ###


* `nvm_types.h` contains type definitions for the most commonly
used types. The most interesting types are:

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


### Kernel module ###


### Typical mode of operation ###

Please refer to section 7 of the NVM Express specification.


References
------------------------------------------------------------------------------
[NVMe1.3]: http://nvmexpress.org/wp-content/uploads/NVM-Express-1_3a-20171024_ratified.pdf
[SmartIO]: http://dolphinics.com/products/pcie_smart_io_device_lending.html
[SISCI]: http://ww.dolphinics.no/download/ci/docs-master/
[PCIe P2P]: https://www.dolphinics.com/download/WHITEPAPERS/Dolphin_Express_IX_Peer_to_Peer_whitepaper.pdf
[Device Lending]: http://dolphinics.com/download/WHITEPAPERS/PCI_Express_device_lending_may_2016.pdf
[SPDK]: http://www.spdk.io/
[3D XPoint]: https://en.wikipedia.org/wiki/3D_XPoint
[GPUDirect]: http://docs.nvidia.com/cuda/gpudirect-rdma/index.html
[GPUDirect Async]: http://on-demand.gputechconf.com/gtc/2016/presentation/s6264-davide-rossetti-GPUDirect.pdf
