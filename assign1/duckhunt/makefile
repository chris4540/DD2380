CC=g++
INCLUDE=-I../util -I../HMM_Model -I./
# CFLAGS=-g -O2 -std=gnu++14 $(INCLUDE)
CXXFLAGS = -Wall -std=gnu++14 $(INCLUDE)
# TARGET=test_models Skeleton
TARGET=Skeleton

all: $(TARGET)

Skeleton: ../HMM_Model/hmm.o main.o Player.o GameServer.o Client.o
	$(CC) -o $@ $^ $(CXXFLAGS)

# TEST for models
# test_models: test/test_models.o ../HMM_Model/hmm.o
# 	$(CC) -o $@ $^ $(CXXFLAGS)

%.o: %.cpp
	$(CC) -c -o $@ $<  $(CXXFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGET) *.o

.PHONY: run
run:
	rm -f /tmp/player2server /tmp/server2player
	mkfifo /tmp/player2server /tmp/server2player
	./Skeleton server < /tmp/player2server | ./Skeleton verbose > /tmp/player2server

.PHONY: run2
run2:
	rm -f /tmp/player2server /tmp/server2player
	mkfifo /tmp/player2server /tmp/server2player
	./Skeleton server load ParadiseEmissions.in < /tmp/player2server | ./Skeleton verbose > /tmp/player2server
