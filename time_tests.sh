#!/bin/bash

# Variables
INPUT_IMAGE="runway.pgm"
OUTPUT_IMAGE_BASE="output_base.png"
OUTPUT_IMAGE_CONST="output_const.png"
OUTPUT_IMAGE_SHARED="output_shared.png"
NUM_RUNS=5

# Function to run an executable multiple times and collect execution times
run_executable() {
    local exe=$1
    local input=$2
    local output=$3
    local -n times_array=$4  # Use nameref for passing array by reference

    for i in $(seq 1 $NUM_RUNS); do
        echo "Running $exe (Run #$i):"
        # Execute the program and capture its output
        output_log=$($exe "$input" --output "$output")

        # Extract the execution time using grep and awk
        time=$(echo "$output_log" | grep "Kernel time execution (ms):" | awk '{print $5}')

        # Check if time was successfully extracted
        if [[ -n "$time" ]]; then
            echo "Execution time: $time ms"
            times_array+=("$time")
        else
            echo "Failed to extract execution time."
            times_array+=("N/A")
        fi
    done
}

# Declare arrays to hold execution times
declare -a times_base
declare -a times_const
declare -a times_shared

# Run houghBase.exe
echo "=== Running houghBase.exe ==="
run_executable "./houghBase.exe" "$INPUT_IMAGE" "$OUTPUT_IMAGE_BASE" times_base
echo

# Run houghConstant.exe
echo "=== Running houghConstant.exe ==="
run_executable "./houghConstant.exe" "$INPUT_IMAGE" "$OUTPUT_IMAGE_CONST" times_const
echo

# Run houghShared.exe
echo "=== Running houghShared.exe ==="
run_executable "./houghShared.exe" "$INPUT_IMAGE" "$OUTPUT_IMAGE_SHARED" times_shared
echo

# Function to calculate average of an array of numbers
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
printf "%-20s %-10s %-10s %-10s %-10s %-10s\n" "Executable" "Run1(ms)" "Run2(ms)" "Run3(ms)" "Run4(ms)" "Run5(ms)" "Average(ms)"
printf "<--------------------------------------------------------------------------------------->\n"

# Function to print a row in the table
print_table_row() {
    local exe_name=$1
    shift
    local times=("$@")
    printf "%-20s" "$exe_name"
    for t in "${times[@]}"; do
        printf " %-10s" "$t"
    done
    printf " %-10s\n" "${!6}"
}

# Print each row
printf "%-20s %-10s %-10s %-10s %-10s %-10s %-10s\n" "houghBase.exe" "${times_base[@]}" "$avg_base"
printf "%-20s %-10s %-10s %-10s %-10s %-10s %-10s\n" "houghConstant.exe" "${times_const[@]}" "$avg_const"
printf "%-20s %-10s %-10s %-10s %-10s %-10s %-10s\n" "houghShared.exe" "${times_shared[@]}" "$avg_shared"

echo "Test execution complete. Check output images."