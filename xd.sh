#!/bin/bash

# Variables
INPUT_IMAGE="runway.pgm"
OUTPUT_IMAGE_BASE="output_base.png"
OUTPUT_IMAGE_CONST="output_const.png"

# Function to calculate mean
calculate_mean() {
    local sum=0
    local count=0

    # Read each value from stdin
    while read -r value; do
        sum=$(echo "$sum + $value" | bc -l)
        ((count++))
    done

    # Calculate and print mean with 3 decimal places
    if [ $count -gt 0 ]; then
        echo "scale=3; $sum / $count" | bc -l
    else
        echo "0"
    fi
}

# Number of iterations
ITERATIONS=5

echo "Running timing tests ($ITERATIONS iterations)..."
echo "----------------------------------------"

# Arrays to store times
declare -a base_times
declare -a const_times

# Run houghBase tests
echo "Testing houghBase.exe:"
for i in $(seq 1 $ITERATIONS); do
    echo "  Iteration $i:"
    # Use time command with format that only outputs seconds
    time_result=$( { time ./houghBase.exe $INPUT_IMAGE --output $OUTPUT_IMAGE_BASE; } 2>&1 )
    # Extract real time and store it
    real_time=$(echo "$time_result" | grep "real" | awk '{print $2}')
    echo "    Time: $real_time seconds"
    base_times+=($real_time)
done

echo ""

# Run houghConstant tests
echo "Testing houghConstant.exe:"
for i in $(seq 1 $ITERATIONS); do
    echo "  Iteration $i:"
    # Use time command with format that only outputs seconds
    time_result=$( { time ./houghConstant.exe $INPUT_IMAGE --output $OUTPUT_IMAGE_CONST; } 2>&1 )
    # Extract real time and store it
    real_time=$(echo "$time_result" | grep "real" | awk '{print $2}')
    echo "    Time: $real_time seconds"
    const_times+=($real_time)
done

echo ""
echo "Results Summary"
echo "----------------------------------------"

# Calculate and display means
echo "houghBase.exe mean execution time: $(printf '%s\n' "${base_times[@]}" | calculate_mean) seconds"
echo "houghConstant.exe mean execution time: $(printf '%s\n' "${const_times[@]}" | calculate_mean) seconds"

echo ""
echo "Test execution complete. Check output images."
