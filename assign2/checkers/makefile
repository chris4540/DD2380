CC=g++
INCLUDE=-I./
CXXFLAGS=-g -O2 -static -Wall -std=gnu++14 $(INCLUDE)
TARGET=checkers
TEST_TARGET=test_zobrist

all: $(TARGET)

test: $(TEST_TARGET)

checkers: gamestate.o main.o player.o
	$(CC) -o $@ $^ $(CXXFLAGS)

test_zobrist: test_zobrist.o gamestate.o
	$(CC) -o $@ $^ $(CXXFLAGS)

test_zobrist.o: test_zobrist.cpp zobrist.hpp
	$(CC) -c -o $@ $< $(CXXFLAGS)

%.o: %.cpp
	$(CC) -c -o $@ $<  $(CXXFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGET) $(TEST_TARGET) *.o

.PHONY: run
run:
	rm -f /tmp/pipe
	mkfifo /tmp/pipe
	./checkers init verbose < /tmp/pipe | ./checkers > /tmp/pipe
