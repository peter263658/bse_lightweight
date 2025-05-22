#!/bin/bash

# Set paths to model checkpoint and test data
MODEL_CHECKPOINT="/raid/R12K41024/LBCCN/outputs/2025-05-21/14-10-34/logs/lightning_logs/version_0/checkpoints/last.ckpt"  # Adjust path as needed
INPUT_FILE="/raid/R12K41024/LBCCN/Dataset/noisy_testset/snr_0dB/20_20-5360_az55_snr+0.0.wav"  # Choose a single file from test set
OUTPUT_DIR="./benchmark_results"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run benchmark
echo "Running BCCTN benchmark on CPU with a single 2-second pattern"
echo "============================================================="

python benchmark_lbccn.py \
    --model_checkpoint $MODEL_CHECKPOINT \
    --input_file $INPUT_FILE \
    --output_dir $OUTPUT_DIR \
    --num_runs 100

echo ""
echo "Benchmark complete. Results saved to $OUTPUT_DIR"
echo "See $OUTPUT_DIR/benchmark_results.txt for detailed results"
