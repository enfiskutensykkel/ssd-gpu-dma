PROJECT	:= cuda-nvme
OBJECTS := userspace/cunvme.c.o
RELEASE := $(shell uname -r)
CUHOME	:= /usr/local/cuda
MODULE	:= cunvme
DEFINES	:= -DCUNVME_FILE='"$(MODULE)"' -DCUNVME_VERSION='"0.1"'


obj-m := $(MODULE).o
$(MODULE)-objs := module/cunvme.o
ccflags-y += -I$(PWD)/include $(DEFINES)
KDIR ?= /lib/modules/$(RELEASE)/build

ifeq ($(KERNELRELEASE),)
	CC    	:= $(CUHOME)/bin/nvcc
	CCBIN	:= /usr/bin/gcc
	CFLAGS	:= -Wall -Wextra -pedantic
	INCLUDE	:= -I$(CUHOME)/include -Iinclude
	LDLIBS	:= -lcuda
	LDFLAGS	:= -L$(CUHOME)/lib64
endif


.PHONY: default reload unload load

default: modules $(PROJECT)

clean:
	-$(RM) $(PROJECT) $(OBJECTS)
	$(MAKE) -C $(KDIR) M=$(PWD) clean

$(PROJECT): $(OBJECTS)
	$(CC) -ccbin $(CCBIN) -o $@ $^ $(LDFLAGS) $(LDLIBS)

reload: unload load

unload:
	-rmmod $(MODULE).ko

load:
	insmod $(MODULE).ko

userspace/%.c.o: userspace/%.c
	$(CCBIN) -std=c99 $(CFLAGS) $(DEFINES) $(INCLUDE) -o $@ $< -c

userspace/%.cu.o: userspace/%.cu
	$(CC) -Xcompiler "$(CFLAGS) $(DEFINES)" $(INCLUDE) -o $@ $< -c

%:
	$(MAKE) -C $(KDIR) M=$(PWD) $@
