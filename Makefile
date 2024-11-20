# Makefile para Hough Transform con CUDA

# Variables
NVCC = nvcc
GCC = g++
CFLAGS = -O3 -I./common
NVCCFLAGS = -O3 -I./common
TARGET_BASE = houghBase.exe
TARGET_CONST = houghConstant.exe
TARGET_SHARED = houghShared.exe
TARGET_SHARED = houghShared.exe
OBJ = pgm.o image_utils.o

# Regla por defecto
all: $(TARGET_BASE) $(TARGET_CONST) $(TARGET_SHARED)
all: $(TARGET_BASE) $(TARGET_CONST) $(TARGET_SHARED)

$(TARGET_BASE): houghBase.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) houghBase.cu $(OBJ) -o $(TARGET_BASE)


$(TARGET_CONST): houghConstant.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) houghConstant.cu $(OBJ) -o $(TARGET_CONST)

$(TARGET_SHARED): houghShared.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) houghShared.cu $(OBJ) -o $(TARGET_SHARED)

$(TARGET_SHARED): houghShared.cu $(OBJ)
	$(NVCC) $(NVCCFLAGS) houghShared.cu $(OBJ) -o $(TARGET_SHARED)

# Regla para pgm.o
pgm.o: common/pgm.cpp common/pgm.h
	$(GCC) $(CFLAGS) -c common/pgm.cpp -o pgm.o

# Regla para image_utils.o
image_utils.o: common/image_utils.cpp common/image_utils.h common/stb_image_write.h
	$(GCC) $(CFLAGS) -c common/image_utils.cpp -o image_utils.o

# Limpieza
clean:
	rm -f $(TARGET_BASE) $(TARGET_CONST) $(TARGET_SHARED) pgm.o image_utils.o
