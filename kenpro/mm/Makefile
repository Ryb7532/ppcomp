CC = gcc
CFLAGS = -O2 -g
LDFLAGS = 
LIBS = 

APP = mm
OBJS = $(APP).o

all: $(APP)

$(APP): $(OBJS)
	$(CC) $^ $(LIBS) -o $@ $(LDFLAGS)

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $*.o

clean:
	rm -f *.o
	rm -f $(APP)
	rm -f *~
