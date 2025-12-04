#!/bin/bash

# Script to test factor fitting for Qwen models at different sizes and batch sizes
# Generates CSV files with timing results for successful runs
# Uses accelerate launch with FSDP enabled

# Configuration
MODEL_SIZES=("32B"  "14B" "7B" "3B" "1.5B" "0.5B")
NUM_TRAIN_SAMPLES=1000

# Number of GPUs to use (adjust as needed)
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
echo "Detected $NUM_GPUS GPUs"

# Batch sizes to test for each model (from largest to smallest)
# Starting with larger batches and working down to find what fits
declare -A BATCH_SIZES
BATCH_SIZES["0.5B"]="32 16 8 4"
BATCH_SIZES["1.5B"]="16 8 4 2"
BATCH_SIZES["3B"]="8 4 2 1"
BATCH_SIZES["7B"]="4 2 1"
BATCH_SIZES["14B"]="2 1"
BATCH_SIZES["32B"]="1"

# Output directory for results
OUTPUT_DIR="./batch_test_results"
mkdir -p "$OUTPUT_DIR"

# Log file for tracking runs
LOG_FILE="$OUTPUT_DIR/batch_test.log"
SUMMARY_FILE="$OUTPUT_DIR/summary.csv"

# Initialize summary CSV
echo "model_size,batch_size,status,csv_file,error_message" > "$SUMMARY_FILE"

echo "Starting batch size testing for Qwen models..." | tee -a "$LOG_FILE"
echo "Results will be saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Configuration:" | tee -a "$LOG_FILE"
echo "  - Using accelerate launch with $NUM_GPUS GPU(s)" | tee -a "$LOG_FILE"
echo "  - FSDP enabled: Yes" | tee -a "$LOG_FILE"
echo "  - Training samples: $NUM_TRAIN_SAMPLES" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# Function to test a specific model size and batch size
test_model_batch() {
    local model_size=$1
    local batch_size=$2
    local factors_name="qwen_${model_size}_bs${batch_size}"
    local csv_file="$OUTPUT_DIR/timing_factors_${model_size}_bs${batch_size}.csv"

    echo "" | tee -a "$LOG_FILE"
    echo "Testing: Model=$model_size, Batch=$batch_size" | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"


    # Run the factor fitting with accelerate launch and FSDP
    if accelerate launch --num_processes="$NUM_GPUS" -m examples.qwen_scaling.fit_factors \
        --model_size "$model_size" \
        --factors_name "$factors_name" \
        --num_train_samples "$NUM_TRAIN_SAMPLES" \
        --factor_batch_size "$batch_size" \
        --use_fsdp \
        --output_csv "$csv_file" 2>&1 | tee -a "$LOG_FILE"; then

        echo "SUCCESS: Model=$model_size, Batch=$batch_size" | tee -a "$LOG_FILE"
        echo "$model_size,$batch_size,SUCCESS,$csv_file," >> "$SUMMARY_FILE"

        return 0
    else
        local exit_code=$?
        echo "FAILED: Model=$model_size, Batch=$batch_size (exit code: $exit_code)" | tee -a "$LOG_FILE"

        # Try to extract error message from log
        local error_msg=$(tail -n 5 "$LOG_FILE" | tr '\n' ' ' | sed 's/,/;/g')
        echo "$model_size,$batch_size,FAILED,,$error_msg" >> "$SUMMARY_FILE"

        return 1
    fi
}

# Main loop - test each model size with different batch sizes
for model_size in "${MODEL_SIZES[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Testing Model Size: $model_size" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    # Get batch sizes for this model
    batch_sizes_str="${BATCH_SIZES[$model_size]}"
    IFS=' ' read -ra batch_sizes <<< "$batch_sizes_str"

    success_count=0
    fail_count=0

    for batch_size in "${batch_sizes[@]}"; do
        if test_model_batch "$model_size" "$batch_size"; then
            ((success_count++))
        else
            ((fail_count++))
            # If we hit OOM, smaller batch sizes might work, so continue
            echo "Continuing to test smaller batch sizes..." | tee -a "$LOG_FILE"
        fi
    done

    echo "" | tee -a "$LOG_FILE"
    echo "Model $model_size summary: $success_count successful, $fail_count failed" | tee -a "$LOG_FILE"
done

# Final summary
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "TESTING COMPLETE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Summary file: $SUMMARY_FILE" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "CSV files location: $OUTPUT_DIR/*.csv" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Show summary
echo "Results Summary:" | tee -a "$LOG_FILE"
cat "$SUMMARY_FILE" | column -t -s ',' | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Total successful runs: $(grep -c "SUCCESS" "$SUMMARY_FILE" || echo 0)" | tee -a "$LOG_FILE"
echo "Total failed runs: $(grep -c "FAILED" "$SUMMARY_FILE" || echo 0)" | tee -a "$LOG_FILE"
