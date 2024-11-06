#!/bin/bash

# Variables
INPUT_IMAGE="runway.pgm"
OUTPUT_IMAGE_BASE="output_base.png"
OUTPUT_IMAGE_CONST="output_const.png"
# THRESHOLD=150.0

# Ejecutar houghBase.exe
echo "Test 1:"
./houghBase.exe $INPUT_IMAGE --output $OUTPUT_IMAGE_BASE 

# Ejecutar houghConstant.exe
echo "Test 2:"
./houghConstant.exe $INPUT_IMAGE --output $OUTPUT_IMAGE_CONST

# Fin del script
echo "Test execution complete. Check output images."
