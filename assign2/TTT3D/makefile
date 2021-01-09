CC=g++
INCLUDE=-I./
CXXFLAGS=-g -O0 -Wall -std=gnu++14 $(INCLUDE)
TARGET=TTT3D

all: $(TARGET)

TTT3D: gamestate.o main.o player.o
	$(CC) -o $@ $^ $(CXXFLAGS)

%.o: %.cpp
	$(CC) -c -o $@ $<  $(CXXFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGET) *.o

.PHONY: run
run:
	rm -f /tmp/pipe
	mkfifo /tmp/pipe
	./TTT3D init verbose < /tmp/pipe | ./TTT3D > /tmp/pipe
