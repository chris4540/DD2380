CC=g++
LIB_FLAGS=-L../util-matrix/ -lutil
# INCLUDE=-I../simple-matrix/src
INCLUDE=-I../util-matrix/
CFLAGS=-g -O2 -static -std=gnu++14 $(LIB_FLAGS) $(INCLUDE)

TARGET=main.out

all: $(TARGET)

main.out: main.o
	$(CC) -o $@ $^ $(CFLAGS)

%.o: %.cpp
	$(CC) -c -o $@ $<  $(CFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGET) *.o

.PHONY: run
run:
	./main.out < sample/sample_00.in
