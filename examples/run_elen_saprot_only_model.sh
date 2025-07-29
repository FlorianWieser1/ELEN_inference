#!/bin/bash
#SBATCH -J ELEN_saprot_only
#SBATCH -o logs/ELEN_saprot_only_%j.log
#SBATCH -e logs/ELEN_saprot_only_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

###############################################################################
# ELEN Example Script - SaProt-Only Mode
# Author: Florian Wieser
# Date: 2025-07-29
#
# Description:
#   Runs ELEN in "saprot_only" feature mode on example input data.
#   This mode uses only sequence (SaProt) features for model quality assessment,
#   without geometric or Rosetta-based features.
#
# Usage:
#   run_elen_saprot_only_model.sh
#
# Requirements:
#   - SLURM cluster with GPU support
#   - Conda environment 'elen_test' activated, with all ELEN dependencies installed
#   - Input directory: input_ELEN
#   - ELEN model weights directory: ../models/
#
# SaProt installation:
# 	SaProt embeddings can be provided either by --saprot_embeddings_file in .h5 format, or computed
# 	at runtime, if SaProt installation is provided (PATH_SAPROT in elen/config.py).
#
# Output:
#   Results are written to output_ELEN_saprot_only_model/
###############################################################################

# Activate conda environment
source activate elen_test

# Configuration
INPUT_DIR="input_ELEN"
OUTPUT_DIR="output_ELEN_saprot_only_model"
MODEL_DIR="../models"

# Run ELEN inference
python ../elen/inference/run_elen_inference.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --pocket_type "RP" \
    --elen_models_dir "$MODEL_DIR" \
    --feature_mode "saprot_only" \
    --overwrite
