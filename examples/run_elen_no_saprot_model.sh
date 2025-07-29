#!/bin/bash
#SBATCH -J ELEN_no_saprot
#SBATCH -o logs/ELEN_no_saprot_%j.log
#SBATCH -e logs/ELEN_no_saprot_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

###############################################################################
# ELEN Example Script - No SaProt (Sequence) Features
# Author: Florian Wieser
# Date: 2025-07-29
#
# Description:
#   Runs ELEN in "no_saprot" feature mode on example input data.
#   This mode uses only geometry and Rosetta-based features,
#   without SaProt (sequence/LLM) features.
#
# Usage:
#   sbatch run_elen_no_saprot_model.sh
#
# Requirements:
#   - SLURM cluster with GPU support
#   - Conda environment 'elen_test' activated, with all ELEN dependencies installed
#   - Input directory: input_ELEN
#   - ELEN model weights directory: ../models/
#
# Output:
#   Results are written to output_ELEN_no_saprot/
###############################################################################

# Activate conda environment
source activate elen_test

# Configuration
INPUT_DIR="input_ELEN"
OUTPUT_DIR="output_ELEN_no_saprot"
MODEL_DIR="../models"

# Run ELEN inference (geometry + Rosetta features only, no SaProt sequence features)
python ../elen/inference/run_elen_inference.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --pocket_type "RP" \
    --elen_models_dir "$MODEL_DIR" \
    --feature_mode "no_saprot" \
    --overwrite