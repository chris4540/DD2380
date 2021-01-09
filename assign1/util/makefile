CC=g++
# CFLAGS=-g -O2 -static -std=gnu++14
CFLAGS=-g -O0 -static -std=gnu++14
LIBRARY=libutil.a
TARGET=test_matrix test_ndarray

all: $(TARGET) $(LIBRARY)

libutil.a: matrix.o
	ar rvs $@ $^

test_matrix: test_matrix.o matrix.o
	$(CC) -o $@ $^ $(CFLAGS)

test_ndarray: test_ndarray.o ndarray.hpp
	$(CC) -o $@ $< $(CFLAGS)

%.o: %.cpp
	$(CC) -c -o $@ $<  $(CFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGET) *.o
