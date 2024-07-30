CC=icx
FC=ifx
CXX=icx
CFLAGS = -O2 -Wall -Wextra -Wshadow -pedantic
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic -fPIC
FCFLAGS  = -O2 -fPI -i8
LIBS=-lifcore -lifport -lstdc++
LIBS_CUDA=-lcusolver -lcublas -lcudart
MOD_DIR = ./mod

.PHONY: libs cuda all clean

libs: accel_lib.o libaccel.so
cuda: libaccel_cuda.so
all: test.x test_c.x cuda

test.x: test.o libaccel.so
	$(FC) $(FCFLAGS) -Wl,--disable-new-dtags -Wl,--rpath=. -L. -o test.x test.o  -laccel  $(LIBS)

test.o: test.f90 accel_lib.o
	$(FC) $(FCFLAGS) -c test.f90 -J$(MOD_DIR)

test_c.x: test_c.o libaccel.so
	$(CC) $(CFLAGS) -Wl,--disable-new-dtags -Wl,--rpath=. -L. -o test_c.x test_c.o -laccel -lm $(LIBS)

test_c.o: test_c.c
	$(CC) $(CFLAGS) -c test_c.c

accel_lib.o: accel_lib.f90
	mkdir $(MOD_DIR)
	$(FC) $(FCFLAGS) -c accel_lib.f90 -o accel_lib.o -J$(MOD_DIR)

libaccel_cuda.o: libaccel_cuda.cpp libaccel_cuda.h
	$(CXX) $(CXXFLAGS) -c -fPIC -o libaccel_cuda.o libaccel_cuda.cpp

libaccel_cuda.so: libaccel_cuda.o
	$(CXX) $(CXXFLAGS) -shared -o libaccel_cuda.so libaccel_cuda.o $(LIBS_CUDA)

libaccel.o: libaccel.cpp libaccel.h
	$(CXX) $(CXXFLAGS) -c -fPIC -o libaccel.o libaccel.cpp

libaccel.so: libaccel.o accel_lib.o
	$(CXX) $(CXXFLAGS) -shared -o libaccel.so libaccel.o accel_lib.o -ldl

clean:
	rm -rf *.mod *.o *.so *.x $(MOD_DIR)
