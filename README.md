libnvm: An API for building userspace NVMe drivers and storage applications
===============================================================================
This library is a userspace API implemented in C for writing custom NVM Express
([NVMe]) drivers and high-performance storage applications. The API provides 
simple semantics and functions which a userspace program can use to control or 
manage one or more NVMe disk controllers.

The API is in essence similar to [SPDK], in that it moves driver code to 
userspace and relies on hardware polling rather than being interrupt driven. 
By mapping userspace memory directly, `libnvm` eliminates the cost of context
switching into kernel space and enables zero-copy access from userspace. 
This greatly reduces the latency of IO operations compared to accessing storage
devices through normal file system abstractions provided by the Linux kernel.

`libnvm` is able to provide a simple low-level block-interface with extremely
low latency in the IO path. With minimal driver support, it is possible to set
up arbitrary memory mappings to device memory, enabling direct IO between NVMe
storage devices and other PCIe devices ([PCIe peer-to-peer]). 

As NVMe is designed in a way that reflects the inherent parallelism in modern
computing architectures, we are able to provide a lock-less interface to the 
disk which can be shared by multiple computing instances. `libnvm` can be 
linked with CUDA programs, enabling **high-performance storage access directly 
from your CUDA kernels**. This is achieved by placing IO queues and data 
buffers directly in GPU memory, **eliminating the need to involve the CPU in 
the IO path entirely**.

A huge benefit of the parallel design of NVMe combined with the possibility of
using arbitrary memory addresses for buffers and queues also means that a disk
can be shared concurrently by multiple computing instances. By setting up 
mappings using a PCIe Non-Transparent Bridge ([PCIe NTB]), it is possible for
**multiple PCIe root complexes to share a disk concurrently**.
The API can be linked with applications using the SISCI  [SmartIO] API from 
Dolphin Interconnect Solutions, allowing the  user to create powerful custom 
configurations of remote and local devices and NVMe disks in a PCIe cluster. 
In other words, it enables concurrent low-latency access to NVMe disks from 
multiple machines in the cluster.





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

For using `libnvm` with your CUDA programs, you need the following:
* An Nvidia GPU capable of [GPUDirect RDMA] and [GPUDirect Async]
  This means either a Quadro or Tesla workstation model using the Kepler 
  architecture or newer.
* An architecture that supports [PCIe peer-to-peer], for example the Intel Xeon
  family of processors. This is strictly required if you are using SmartIO or
  plan on using RDMA.
* The _FindCUDA_ package for CMake.
* GCC version 5.4.0 or newer. Compiler must be able to compile C++11 and 
  POSIX threads.
* CUDA 8.0 or newer with CUDA development toolkit.
* Kernel module symbols and headers for your Nvidia driver.

For linking with [SISCI] API, you additionally need the Dolphin 5.5.0 software
base (or newer) with CUDA support and SmartIO enabled. 



### Disable IOMMU ###
If you are using CUDA or implementing support for your own custom devices, 
you need to explicitly disable IOMMU as IOMMU support for peer-to-peer on 
Linux is a bit flaky at the moment. If you are not relying on peer-to-peer,
I would in fact recommend you leaving the IOMMU _on_ for protecting memory 
from rogue writes.

To check if the IOMMU is on, you can do the following:
```
$ cat /proc/cmdline | grep iommu
```
If either `iommu=on` or `intel_iommu=on` is found by `grep`, the IOMMU
is enabled.

You can disable it by removing `iommu=on` and `intel_iommu=on` from the 
`CMDLINE` variable in `/etc/default/grub` and then reconfiguring GRUB.
The next time you reboot, the IOMMU will be disabled.

As soon as peer-to-peer IOMMU support is improved in the Linux API and the
Nvidia driver supports it, I will add it to the kernel module.


### Using CUDA without SmartIO ####
If you are going to use CUDA, you also need to locate the kernel module
directory and manually run `make`. Locations will vary on different distros
and based on installation type, but on Ubuntu the driver source can be usually found 
in `/usr/src/nvidia-<major>-<major>.<minor>` if you install CUDA through the
`.deb.` package.

The CMake configuration is _supposed to_ autodetect the location of CUDA, 
and the Nvidia driver by looking for a file called `Module.symvers` in known directories. 
Make sure that this file is generated. It is also possible to point CMake to the correct
location of the driver by specifying the `NVIDIA` define

Make sure that the output from CMake contains both `Using NVIDIA driver found in ...` and `Configuring kernel module with CUDA`.


### Building the project ###
From the project root directory, do the following:
```
$ mkdir -p build; cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release # use =Debug for debug build
$ make libnvm                         # builds library
$ make examples                       # builds example programs
```

The CMake configuration is _supposed to_ autodetect the location of CUDA, 
Nvidia driver and SISCI library. CUDA is located by the _FindCUDA_ package for
CMake, while the location of both the Nvidia driver and SISCI can be manually
set by overriding the `NVIDIA` and `DIS` defines for CMake 
(`cmake .. -DNVIDIA=/usr/src/...` -DDIS=/opt/DIS/`).

After this, you should also compile the `libnvm` helper kernel module unless
you are using SISCI SmartIO. Assuming that you are still standing in the build
directory, do the following:
```
$ cd module; make # only required if not using SISCI SmartIO
```

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
$ /opt/DIS/sbin/smartio_tool add 05:00.0
$ /opt/DIS/sbin/smartio_tool available 05:00.0
$
$ # Find out the local node identifier
$ /opt/DIS/sbin/dis_config -gn
Card 1 - NodeId:  8
$
$ # Connect to the local node
$ /opt/DIS/sbin/smartio_tool connect 8
$
$ # Find out the device identifier
$ /opt/DIS/sbin/smartio_tool list
80000: Non-Volatile memory controller Intel Corporation Device f1a5 [available]
$
$ # Build library and identify example
$ make libnvm && make identify-smartio
$
$ ./bin/nvm-identify --ctrl=0x80000  # use the device id
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



### Using the libnvm helper kernel module ###
If you are not using SISCI SmartIO, you must use the project's kernel module
in order to map GPU memory for the NVMe disk.
Currently the only version of Linux tested is Linux 4.11.0. Other versions
may work, but you probably have to change the call to `get_user_pages()`
as well as any calls to the DMA API.

Repeating the requirements from the section above, you should make sure that
you use a processor that supports [PCIe peer-to-peer], and that you have a GPU 
with [GPUDirect] support. Remember to [disable the IOMMU](#disable-iommu).
If you are not using CUDA (or any other third-party stuff), it is recommended 
that you leave the IOMMU on.

Loading and unloading the driver is done as follows:
```
$ cd build/module
$ make
$ make load     # will insert the kernel module
$ make unload   # unloads the kernel module
```

You want to unload the default nvme driver for the NVMe disk, and bind 
the helper driver to it:
```
$ echo -n "0000:05:00.0" > /sys/bus/pci/devices/0000\:05\:00.0/driver/unbind
$ echo -n "0000:05:00.0" > /sys/bus/pci/drivers/libnvm\ helper/bind
```

After doing this, the file `/dev/libnvm0` should show up, representing the
disk's BAR0.



### All CMake build settings ###
Settings can be passed to CMake using the `-Dsetting=value` flag. Here is a 
comprehensive list of settings that can be overridden.

Setting                 | Default        | Explanation
------------------------|----------------|-----------------------------------------
`CMAKE_BUILD_TYPE`      | `Debug`        | Set to `Release` to make a release build
`DIS`	                | `/opt/DIS`     | Override the Dolphin installation path
`NVIDIA`                |                | Override path to Nvidia driver
`nvidia_archs`		| 30;50;60;61;70 | Specify compute modes and SMs
`no_smartio`            | `false`        | Don't build API with SmartIO support
`no_module`             | `false`        | Don't build kernel module
`no_cuda`               | `false`        | Don't build API with CUDA support
`no_smartio_samples`    | `false`        | Don't build SmartIO samples
`no_smartio_benchmarks` | `false`        | Don't build SmartIO benchmarks





Non-Volatile Memory Express (NVMe)
-------------------------------------------------------------------------------
[NVMe] is a software specification for disk controllers (_drives_) that 
provides storage on non-volatile media, for example flash memory or Intel's
[3D XPoint].

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









Nvidia GPUDirect
------------------------------------------------------------------------------

Programs intended for running on GPUs or other computing accelerators that 
support Remote DMA (RDMA), can use this library to enable direct disk access
from the accelerators. Currently, the library supports setting up mappings
for GPUDirect-capable Nvidia GPUs. 



PCIe NTBs and Dolphin SmartIO
------------------------------------------------------------------------------
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






API overview
------------------------------------------------------------------------------



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



[NVMe]: http://nvmexpress.org/wp-content/uploads/NVM-Express-1_3a-20171024_ratified.pdf
[SmartIO]: http://dolphinics.com/products/pcie_smart_io_device_lending.html
[SISCI]: http://ww.dolphinics.no/download/SISCI_DOC_V2/index.html
[PCIe NTB]: http://www.dolphinics.com/products/pcie_Intel_NTB_networking.html
[PCIe peer-to-peer]: https://www.dolphinics.com/download/WHITEPAPERS/Dolphin_Express_IX_Peer_to_Peer_whitepaper.pdf
[Device Lending]: http://dolphinics.com/download/WHITEPAPERS/PCI_Express_device_lending_may_2016.pdf
[SPDK]: http://www.spdk.io/
[3D XPoint]: https://en.wikipedia.org/wiki/3D_XPoint
[GPUDirect]: https://developer.nvidia.com/gpudirect
[GPUDirect RDMA]: http://docs.nvidia.com/cuda/gpudirect-rdma/index.html
[GPUDirect Async]: http://on-demand.gputechconf.com/gtc/2016/presentation/s6264-davide-rossetti-GPUDirect.pdf
