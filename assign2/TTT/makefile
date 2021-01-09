CC=g++
INCLUDE=-I./
CXXFLAGS=-g -O0 -Wall -std=gnu++14 $(INCLUDE)
TARGET=TTT

all: $(TARGET)

TTT: gamestate.o main.o player.o
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
	./$(TARGET) init verbose < /tmp/pipe | ./$(TARGET) > /tmp/pipe
