TARGET	:= ssd_dma
OBJECTS := src/main.o src/nvme.o
DISHOME := /opt/DIS
RELEASE	:= $(shell uname -r)

obj-m := $(TARGET).o
$(TARGET)-objs := $(OBJECTS)
ccflags-y += -DSSD_DMA_FILE_NAME='"$(TARGET)"' -DSSD_DMA_VERSION='"0.2"' -I$(PWD)/include
KDIR ?= /lib/modules/$(RELEASE)/build

ifeq ($(KERNELRELEASE),)
	CC 	:= gcc
	CFLAGS	:= -Wall -Wextra -pedantic -D_REENTRANT -Wno-unused-parameter -DSSD_DMA_FILE_NAME='"/proc/$(TARGET)"'
	INCLUDE	:= -I$(DISHOME)/include -I$(DISHOME)/include/dis -Iinclude
	LDLIBS	:= -lsisci -lpthread
	LDFLAGS	:= -L$(DISHOME)/lib64
endif

.PHONY: default reload unload load

default: modules example

example: test/example.c
	$(CC) -std=gnu99 -o $@ $(CFLAGS) $(INCLUDE) $(LDFLAGS) $^ $(LDLIBS)

reload: unload load

unload:
	-rmmod $(TARGET).ko

load:
	insmod $(TARGET).ko

clean:
	-$(RM) example
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	
%:
	$(MAKE) -C $(KDIR) M=$(PWD) $@

