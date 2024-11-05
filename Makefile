# Makefile para Hough Transform con CUDA

# Variables
NVCC = nvcc
GCC = g++
CFLAGS = -O3 -I./common
NVCCFLAGS = -O3 -I./common
TARGET = houghBase.exe
OBJ = pgm.o image_utils.o

# Regla por defecto
all: $(TARGET)

# Regla para el ejecutable
$(TARGET): houghBase.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) houghBase.cu $(OBJ) -o $(TARGET)

# Regla para pgm.o
pgm.o: common/pgm.cpp common/pgm.h
	$(GCC) $(CFLAGS) -c common/pgm.cpp -o pgm.o

# Regla para image_utils.o
image_utils.o: common/image_utils.cpp common/image_utils.h common/stb_image_write.h
	$(GCC) $(CFLAGS) -c common/image_utils.cpp -o image_utils.o

# Limpieza
clean:
	rm -f $(TARGET) pgm.o image_utils.o
