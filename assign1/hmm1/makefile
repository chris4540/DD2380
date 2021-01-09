CC=g++
INCLUDE=-I../util -I../HMM_Model
CFLAGS=-g -O2 -static -std=gnu++14 $(INCLUDE)

TARGET=main2

all: $(TARGET)

# main.out: main.o
# 	$(CC) -o $@ $^ $(CFLAGS)

main2: main2.o ../HMM_Model/hmm.o
	$(CC) -o $@ $^ $(CFLAGS)

%.o: %.cpp
	$(CC) -c -o $@ $<  $(CFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGET) *.o

.PHONY: run
run:
	./main.out < samples/hmm2_01.in
