#!/bin/bash

# Variables
INPUT_IMAGE="input/runway.pgm"
OUTPUT_IMAGE_BASE="output/output_base.png"
OUTPUT_IMAGE_CONST="output/output_const.png"
OUTPUT_IMAGE_SHARED="output/output_shared.png"
NUM_RUNS=10  # Changed from 5 to 10

# Create the output directory if it doesn't exist
mkdir -p output

# Function to execute an executable multiple times and collect execution times
run_executable() {
    local exe=$1
    local input=$2
    local output=$3
    local -n times_array=$4  # Use nameref to pass the array by reference

    for i in $(seq 1 $NUM_RUNS); do
        echo "Running $exe (Run #$i):"

        # Check if the executable exists
        if [[ ! -f "$exe" ]]; then
            echo "Error: $exe does not exist."
            times_array+=("N/A")
            continue
        fi

        # Execute the program and capture its output
        output_log=$("$exe" "$input" --output "$output")

        # Extract the execution time using grep and awk
        time=$(echo "$output_log" | grep "Tiempo de Kernel (ms):" | awk '{print $5}')

        # Verify if the time was extracted correctly
        if [[ -n "$time" ]]; then
            echo "Execution time: $time ms"
            times_array+=("$time")
        else
            echo "Failed to extract execution time."
            times_array+=("N/A")
        fi
    done
}

# Declare arrays to store execution times
declare -a times_base
declare -a times_const
declare -a times_shared

# Execute houghBase.exe
echo "=== Running houghBase.exe ==="
run_executable "bin/houghBase.exe" "$INPUT_IMAGE" "$OUTPUT_IMAGE_BASE" times_base
echo

# Execute houghConstant.exe
echo "=== Running houghConstant.exe ==="
run_executable "bin/houghConstant.exe" "$INPUT_IMAGE" "$OUTPUT_IMAGE_CONST" times_const
echo

# Execute houghShared.exe
echo "=== Running houghShared.exe ==="
run_executable "bin/houghShared.exe" "$INPUT_IMAGE" "$OUTPUT_IMAGE_SHARED" times_shared
echo

# Function to calculate the average of an array of numbers
calculate_average() {
    local -n arr=$1
    local sum=0
    local count=0

    for val in "${arr[@]}"; do
        # Ensure the value is a number
        if [[ "$val" != "N/A" ]]; then
            sum=$(echo "$sum + $val" | bc)
            count=$((count + 1))
        fi
    done

    if [[ $count -gt 0 ]]; then
        average=$(echo "scale=2; $sum / $count" | bc)
        echo "$average"
    else
        echo "N/A"
    fi
}

# Calculate averages
avg_base=$(calculate_average times_base)
avg_const=$(calculate_average times_const)
avg_shared=$(calculate_average times_shared)

# Print the results in a table
echo "=== Execution Times ==="
# Header
printf "%-8s %-20s %-20s %-20s\n" "Run" "houghBase (ms)" "houghConstant (ms)" "houghShared (ms)"
printf "<------------------------------------------------------------------------------->\n"

# Print each run's times
for i in $(seq 1 $NUM_RUNS); do
    # Adjust index since arrays are zero-based
    index=$((i-1))
    printf "%-8s %-20s %-20s %-20s\n" "Run$i" "${times_base[index]}" "${times_const[index]}" "${times_shared[index]}"
done

# Print the average row
printf "<------------------------------------------------------------------------------->\n"
printf "%-8s %-20s %-20s %-20s\n" "Average" "$avg_base" "$avg_const" "$avg_shared"

echo "Test execution complete. Check output images."
