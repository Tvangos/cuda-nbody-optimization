#!/bin/bash
# Run code multiple times and save summary to CSV
# Modified to track Time & Throughput, calculating average after excluding Min/Max.
# Usage: ./run_mul_times.sh <version> <threads> <DEBUG>

VERSION=$1
THREADS=$2
DEBUG_FLAG=$3 # 0 or 1
RUNS=12
EXE="${VERSION}/nbody"

export OMP_NUM_THREADS=$THREADS

declare -a throughput_vals
declare -a time_vals

CSV_SUMMARY="${VERSION}/results_${VERSION}.csv"

# Fresh build
make -f Makefile_cuda clean DIR=$VERSION
make -f Makefile_cuda DIR=$VERSION

echo "Running '$EXE' $RUNS times with $THREADS threads..."
echo "-----------------------------------"

for i in $(seq 1 $RUNS); do
    python3 ./generate_dataset.py

    OUTPUT=$(DEBUG=$DEBUG_FLAG $EXE)

    if [ "$DEBUG_FLAG" -eq 1 ]; then
        # Extract success rate (if present)
        SUCCESS_RATE=$(echo "$OUTPUT" | grep "Success Rate:" | sed -n 's/.*Success Rate: \([0-9.]*\)%.*/\1/p')

        if [[ -n "$SUCCESS_RATE" ]]; then
            if (( $(echo "$SUCCESS_RATE < 99.0" | bc -l) )); then
                echo "❌ Correctness FAILED: Success Rate = $SUCCESS_RATE% (< 99%)"
                exit 1
            else
                echo "✅ Correctness OK: Success Rate = $SUCCESS_RATE%"
            fi
        fi
    fi

    # --- Extract Throughput ---
    VAL=$(echo "$OUTPUT" | grep "Average Throughput:" | grep -v "CPU" \
        | sed -n 's/.*Average Throughput: \([0-9.]*\) Billion.*/\1/p')

    # --- Extract Time ---
    TIME_VAL=$(echo "$OUTPUT" | grep "Total Time:" \
        | sed -n 's/.*Total Time: \([0-9.]*\) seconds.*/\1/p')

    if [[ -z "$VAL" ]] || [[ -z "$TIME_VAL" ]]; then
        echo "Error: Could not extract throughput or time."
        exit 1
    fi

    echo "Run $i: $VAL GInt/s | $TIME_VAL s"
    throughput_vals+=($VAL)
    time_vals+=($TIME_VAL)
done

# ==============================================================================
# Helper function to calculate Trimmed Average (excludes Min and Max)
# ==============================================================================
calc_trimmed_stats() {
    local -n arr=$1  # Pass array by reference
    local sorted
    
    # Sort the array numerically
    sorted=($(printf "%s\n" "${arr[@]}" | sort -n))
    
    local count=${#sorted[@]}
    local min=${sorted[0]}
    local max=${sorted[$((count-1))]}

    # We need at least 3 values to trim min and max
    if (( count > 2 )); then
        # Remove the last element (max)
        unset sorted[$((count-1))]
        # Remove the first element (min)
        unset sorted[0]
    fi

    # Calculate Sum of remaining
    local sum=0
    for t in "${sorted[@]}"; do
        sum=$(echo "$sum + $t" | bc -l)
    done

    # Calculate Average (Divide by original count - 2)
    local divisor=$((count - 2))
    if (( divisor < 1 )); then divisor=1; fi # Safety for small run counts
    
    local avg=$(echo "$sum / $divisor" | bc -l)

    # Return string: "AVG MIN MAX"
    echo "$avg $min $max"
}

# ==============================================================================
# Calculate Stats
# ==============================================================================

# Calculate Throughput Stats
read avg_tput min_tput max_tput <<< $(calc_trimmed_stats throughput_vals)

# Calculate Time Stats
read avg_time min_time max_time <<< $(calc_trimmed_stats time_vals)

# ==============================================================================
# Output to CSV and Console
# ==============================================================================

# CSV Headers: Average Throughput, Average Time, (Optional: Min/Max for reference)
echo "avg_throughput,avg_time,min_throughput,max_throughput,min_time,max_time" > "$CSV_SUMMARY"
echo "$avg_tput,$avg_time,$min_tput,$max_tput,$min_time,$max_time" >> "$CSV_SUMMARY"

echo "-----------------------------------"
echo "Summary written to $CSV_SUMMARY"
echo "Note: Averages calculated excluding Min and Max outliers."
printf "Throughput (GInt/s) -> Avg: %.4f | Min: %.4f | Max: %.4f\n" "$avg_tput" "$min_tput" "$max_tput"
printf "Time (seconds)      -> Avg: %.4f | Min: %.4f | Max: %.4f\n" "$avg_time" "$min_time" "$max_time"