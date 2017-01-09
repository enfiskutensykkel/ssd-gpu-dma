CC	:= gcc
CFLAGS	:= -Wall -Wextra -pedantic -D_REENTRANT -Wno-unused-parameter

DISHOME := /opt/DIS

INCLUDE := -I$(DISHOME)/include -I$(DISHOME)/include/dis -Iinclude
LDLIBS	:= -lsisci -lpthread
LDFLAGS	:= -L$(DISHOME)/lib64

.PHONY: all clean module

all: user 

clean:
	-$(RM) client server build/*.o
	$(MAKE) -C module/ clean

module: 
	$(MAKE) -C module/

user: userspace/user.o
	$(CC) -o $@ $^ $(LDFLAGS) $(LDLIBS)

%.o: %.c
	$(CC) -std=gnu99 $(CFLAGS) $(INCLUDE) -o $@ $< -c
