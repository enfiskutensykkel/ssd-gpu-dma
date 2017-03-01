PROJECT	:= usrspc-nvme

FILES	:= $(wildcard src/*.c)
OBJS	:= $(FILES:%.c=%.o)

CC    	:= gcc
CFLAGS	:= -std=gnu99 -Wall -Wextra -pedantic
DEFINES	:= -DPAGE_SIZE=0x1000
INCLUDE	:=
LDLIBS	:=
LDFLAGS	:=


.PHONY: all clean

all: $(PROJECT)

clean:
	-$(RM) $(PROJECT) $(OBJS)

$(PROJECT): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS) $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE) $(DEFINES) -o $@ $< -c
