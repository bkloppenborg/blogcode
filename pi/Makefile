
LIBS=-lOpenCL -lGL
INCLUDES=-I/usr/include -I/usr/local/cuda/include -I../ -I../../
CC=clang++ $(COMPILER_OPTIONS)
COMPILER_OPTIONS=-std=c++11 -g

all: pi.cpp Makefile
	$(CC) pi.cpp -o pi $(INCLUDES) $(LIBS) $(LIB_PATHS)
	./pi
#	optirun ./pi

