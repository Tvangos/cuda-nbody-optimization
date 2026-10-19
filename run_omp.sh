#!/bin/bash
# Run a single version of code 12 times. Calculates average time and stddev
# first argument of program if version to run, second argument is number of thread to be created
# example run: ./run_mul_times.sh omp 32

VERSION=$1
RUNS=12
EXE="${VERSION}/nbody"
THREADS=$2
declare -a times # Define an array to hold the timings

export OMP_NUM_THREADS=$THREADS

echo "Running '$EXE/nbody' $RUNS times with $THREADS threads..."
echo "-----------------------------------"

# Run the loop and store values in the array
for i in $(seq 1 $RUNS); do
    echo -n "Run $i/$RUNS... "
    # Capture output
    OUTPUT=$($EXE)
    
    # Extract the number using sed (Parses: "Total Time: 5.782 seconds")
    # This captures the digits/dots between "Time: " and " seconds"
    VAL=$(echo "$OUTPUT" | grep "Total Time:" | sed -n 's/.*Total Time: \([0-9.]*\) seconds.*/\1/p')
    
    if [[ -z "$VAL" ]]; then
        echo "Error: Could not extract time from run $i"
        exit 1
    fi

    echo "Run $i: ${VAL}s"
    times+=($VAL) # Append to array
done

echo "-----------------------------------"

# Calculate Stats using bc
sum=0
sumsq=0

for t in "${times[@]}"; do
    # Accumulate sum
    sum=$(echo "$sum + $t" | bc -l)
    # Accumulate sum of squares
    sumsq=$(echo "$sumsq + ($t * $t)" | bc -l)
done

# Calculate Average
avg=$(echo "$sum / $RUNS" | bc -l)

# Calculate Standard Deviation
# Formula: sqrt( (sumsq - (sum^2 / N)) / (N - 1) )
# Note: We use N-1 for Sample Standard Deviation
stdev=$(echo "sqrt(( $sumsq - ($sum * $sum / $RUNS) ) / ($RUNS - 1))" | bc -l)

# Print results (formatted to 4 decimal places)
printf "Average: %.4f seconds\n" "$avg"
printf "Stdev:   %.4f seconds\n" "$stdev"