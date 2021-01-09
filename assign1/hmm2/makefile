CC=g++
LIB_FLAGS=-L../util-matrix/ -lutil
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
	./main.out < samples/hmm3_01.in
