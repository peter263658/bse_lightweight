#!/bin/bash

# Set paths
CLEAN_DIR="/raid/R12K41024/Dataset/liprispeech/librespeech"  # Path to Librispeech
NOISE_DIR="/raid/R12K41024/Dataset/DEMAND/demand"                    # Path to DEMAND dataset
HRIR_PATH="/raid/R12K41024/Dataset/HRIR_database_wav/hrir/anechoic"     # Path to HRIR files
OUTPUT_DIR="/raid/R12K41024/LBCCN/Dataset"                   # Base path where to save the dataset

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Check if prepare_data.py exists
if [ ! -f "prepare_data.py" ]; then
    echo "Error: prepare_data.py not found in current directory"
    exit 1
fi

# Create dataset - use vctk type with SNR subdirectories for test set only
echo "Creating LBCCN dataset with Librispeech and DEMAND..."
python prepare_data.py \
    --clean_dir "${CLEAN_DIR}" \
    --noise_dir "${NOISE_DIR}" \
    --hrir_path "${HRIR_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --hrir_format wav \
    --dataset_type vctk \
    --use_snr_subdirs true

echo "Dataset preparation complete!"