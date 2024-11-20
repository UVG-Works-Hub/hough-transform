# Makefile for Hough Transform with CUDA

# =============================
# Compiler and Flags
# =============================

# CUDA Compiler
NVCC := nvcc

# C++ Compiler
CXX := g++

# C++ Compilation Flags
CFLAGS := -O3 -I./common

# CUDA Compilation Flags
NVCCFLAGS := -O3 -I./common

# =============================
# Directories
# =============================

# Directory for binaries and object files
BIN_DIR := ./bin

# Source directories
SRC_DIR := ./src
COMMON_DIR := ./common

# =============================
# Targets and Objects
# =============================

# List of executable targets
TARGETS := \
    $(BIN_DIR)/houghBase.exe \
    $(BIN_DIR)/houghConstant.exe \
    $(BIN_DIR)/houghShared.exe

# List of common object files
COMMON_OBJS := \
    $(BIN_DIR)/pgm.o \
    $(BIN_DIR)/image_utils.o

# =============================
# Phony Targets
# =============================

.PHONY: all clean

# =============================
# Default Target
# =============================

all: $(TARGETS)

# =============================
# Ensure BIN_DIR Exists
# =============================

# Rule to create bin directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# =============================
# Explicit Rules for Executables
# =============================

# Rule for building houghBase executable
$(BIN_DIR)/houghBase.exe: $(SRC_DIR)/houghBase.cu $(COMMON_OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/houghBase.cu $(COMMON_OBJS) -o $@

# Rule for building houghConstant executable
$(BIN_DIR)/houghConstant.exe: $(SRC_DIR)/houghConstant.cu $(COMMON_OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/houghConstant.cu $(COMMON_OBJS) -o $@

# Rule for building houghShared executable
$(BIN_DIR)/houghShared.exe: $(SRC_DIR)/houghShared.cu $(COMMON_OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRC_DIR)/houghShared.cu $(COMMON_OBJS) -o $@

# =============================
# Explicit Rules for Common Object Files
# =============================

# Rule to compile pgm.cpp into pgm.o
$(BIN_DIR)/pgm.o: $(COMMON_DIR)/pgm.cpp $(COMMON_DIR)/pgm.h | $(BIN_DIR)
	$(CXX) $(CFLAGS) -c $(COMMON_DIR)/pgm.cpp -o $@

# Rule to compile image_utils.cpp into image_utils.o
$(BIN_DIR)/image_utils.o: $(COMMON_DIR)/image_utils.cpp $(COMMON_DIR)/image_utils.h $(COMMON_DIR)/stb_image_write.h | $(BIN_DIR)
	$(CXX) $(CFLAGS) -c $(COMMON_DIR)/image_utils.cpp -o $@

# =============================
# Clean Target
# =============================

clean:
	rm -f $(TARGETS) $(COMMON_OBJS)
