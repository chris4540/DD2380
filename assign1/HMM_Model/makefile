CC=g++
INCLUDE=-I../util
CFLAGS=-g -O2 -static -std=gnu++14 $(INCLUDE) -Wall

TARGET=main.out

all: $(TARGET)

main.out: main.o hmm.o
	$(CC) -o $@ $^ $(CFLAGS)

%.o: %.cpp
	$(CC) -c -o $@ $<  $(CFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGET) *.o

.PHONY: run
run:
	@./main.out < samples/hmm4_01.in
	@echo "===================================================================="
	@cat samples/hmm4_01.ans

	@echo ""

	@./main.out < samples/hmm4_02.in
	@echo "===================================================================="
	@cat samples/hmm4_02.ans

	@echo ""

	@./main.out < samples/hmm4_03.in
	@echo "===================================================================="
	@cat samples/hmm4_03.ans
