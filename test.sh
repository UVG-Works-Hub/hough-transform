#!/bin/bash

# Variables
INPUT_IMAGE="input/runway.pgm"
OUTPUT_IMAGE_BASE="output/output_base.png"
OUTPUT_IMAGE_CONST="output/output_const.png"
OUTPUT_IMAGE_SHARED="output/output_shared.png"
NUM_RUNS=10

# Create the output directory if it doesn't exist
mkdir -p output

# Function to execute an executable multiple times and collect execution times
run_executable() {
    local exe=$1
    local input=$2
    local output=$3
    local -n times_gpu=$4  # GPU times array
    local -n times_cpu_ref=$5  # CPU times array (separate)

    for i in $(seq 1 $NUM_RUNS); do
        echo "Running $exe (Run #$i):"

        # Check if the executable exists
        if [[ ! -f "$exe" ]]; then
            echo "Error: $exe does not exist."
            times_gpu+=("N/A")
            times_cpu_ref+=("N/A")
            continue
        fi

        # Execute the program and capture its output
        output_log=$("$exe" "$input" --output "$output")

        # Extract CPU time
        time_cpu=$(echo "$output_log" | grep "Tiempo de CPU (ms):" | awk '{print $5}')
        if [[ -n "$time_cpu" ]]; then
            echo "CPU Execution time: $time_cpu ms"
            times_cpu_ref+=("$time_cpu")
        else
            echo "Failed to extract CPU time."
            times_cpu_ref+=("N/A")
        fi

        # Extract GPU time
        time_gpu=$(echo "$output_log" | grep "Tiempo de Kernel (ms):" | awk '{print $5}')
        if [[ -n "$time_gpu" ]]; then
            echo "GPU Execution time: $time_gpu ms"
            times_gpu+=("$time_gpu")
        else
            echo "Failed to extract GPU time."
            times_gpu+=("N/A")
        fi
    done
}

# Declare arrays to store execution times
declare -a times_base
declare -a times_const
declare -a times_shared
declare -a times_cpu
declare -a times_cpu_dummy

# Execute houghBase.exe
echo "=== Running houghBase.exe ==="
run_executable "bin/houghBase.exe" "$INPUT_IMAGE" "$OUTPUT_IMAGE_BASE" times_base times_cpu
echo

# Execute houghConstant.exe
echo "=== Running houghConstant.exe ==="
run_executable "bin/houghConstant.exe" "$INPUT_IMAGE" "$OUTPUT_IMAGE_CONST" times_const times_cpu_dummy
echo

# Execute houghShared.exe
echo "=== Running houghShared.exe ==="
run_executable "bin/houghShared.exe" "$INPUT_IMAGE" "$OUTPUT_IMAGE_SHARED" times_shared times_cpu_dummy
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
avg_cpu=$(calculate_average times_cpu)

# Calculate speedups
calculate_speedup() {
    local avg_gpu=$1
    local avg_cpu=$2
    if [[ "$avg_gpu" != "N/A" && "$avg_cpu" != "N/A" && "$avg_cpu" != 0 ]]; then
        echo "scale=2; $avg_cpu / $avg_gpu" | bc
    else
        echo "N/A"
    fi
}

speedup_base=$(calculate_speedup "$avg_base" "$avg_cpu")
speedup_const=$(calculate_speedup "$avg_const" "$avg_cpu")
speedup_shared=$(calculate_speedup "$avg_shared" "$avg_cpu")

# Print the results in a table
echo "=== Execution Times ==="
# Header
printf "%-8s %-20s %-20s %-20s %-20s\n" "Run" "houghBase (ms)" "houghConstant (ms)" "houghShared (ms)" "CPU (ms)"
printf "<---------------------------------------------------------------------------------------->\n"

# Print each run's times
for i in $(seq 1 $NUM_RUNS); do
    # Adjust index since arrays are zero-based
    index=$((i-1))
    printf "%-8s %-20s %-20s %-20s %-20s\n" "Run$i" "${times_base[index]}" "${times_const[index]}" "${times_shared[index]}" "${times_cpu[index]}"
done

# Print the average row
printf "<---------------------------------------------------------------------------------------->\n"
printf "%-8s %-20s %-20s %-20s %-20s\n" "Average" "$avg_base" "$avg_const" "$avg_shared" "$avg_cpu"

# Print the speedup row
printf "<---------------------------------------------------------------------------------------->\n"
printf "%-8s %-20s %-20s %-20s\n" "Speedup" "$speedup_base" "$speedup_const" "$speedup_shared"

echo "Test execution complete. Check output images."
