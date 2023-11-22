CFLAGS = -O2 -Wall -Wextra -Wshadow -pedantic
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic
FCFLAGS  = -O2 -Wall -Wextra -pedantic

.PHONY: libs cuda all clean

libs: accel_lib.o libaccel.so
cuda: libaccel_cuda.so
all: test.x test_c.x cuda

test.x: test.o accel_lib.o libaccel.so
	$(FC) $(FCFLAGS) -fopenmp -Wl,--rpath=. -L. -o test.x test.o accel_lib.o -laccel

test.o: test.f90 accel_lib.o
	$(FC) $(FCFLAGS) -c test.f90

test_c.x: test_c.o libaccel.so
	$(CC) $(CFLAGS)  -Wl,--rpath=. -L. -o test_c.x test_c.o -laccel -lm

test_c.o: test_c.c
	$(CC) $(CFLAGS) -c test_c.c

accel_lib.o: accel_lib.f90
	$(FC) $(FCFLAGS) -c accel_lib.f90

libaccel_cuda.o: libaccel_cuda.cpp libaccel_cuda.h
	$(CXX) $(CXXFLAGS) -c -fPIC -o libaccel_cuda.o libaccel_cuda.cpp

libaccel_cuda.so: libaccel_cuda.o
	$(CXX) $(CXXFLAGS) -shared -o libaccel_cuda.so libaccel_cuda.o -lcusolver -lcublas -lcudart

libaccel.o: libaccel.cpp libaccel.h
	$(CXX) $(CXXFLAGS) -c -fPIC -o libaccel.o libaccel.cpp

libaccel.so: libaccel.o
	$(CXX) $(CXXFLAGS) -shared -o libaccel.so libaccel.o -ldl

clean:
	rm -f *.mod *.o *.so *.x