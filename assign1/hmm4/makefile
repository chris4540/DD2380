CC=g++
INCLUDE=-I../util -I../HMM_Model -I./
CXXFLAGS = -Wall -std=gnu++14 $(INCLUDE)
TARGET=hmm_q7.out hmm_q8.out hmm_q9.out hmm_q10.out hmm_q10iii.out

all: $(TARGET)

%.out: %.o ../HMM_Model/hmm.o
	$(CC) -o $@ $^ $(CXXFLAGS)

%.o: %.cpp
	$(CC) -c -o $@ $<  $(CXXFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGET) *.o

.PHONY: run
run:
	./hmm_q7 < samples/hmm_c_N1000.in