#!/bin/bash
#SBATCH -J ELEN_full
#SBATCH -o logs/ELEN_full_%j.log
#SBATCH -e logs/ELEN_full_%j.err

###############################################################################
# ELEN Example Script - Full Model Inference
# Author: Florian Wieser
# Date: 2025-07-29
# Description:
#   Runs ELEN in full feature mode (geometry + Rosetta + LLM features) on example data.
#
# Usage:
#   run_elen_full_model.sh
#
# Requirements:
#   - Conda environment 'elen_inference' with all dependencies installed
#   - Input directory: input_ELEN_full_model containing .pdb file(s)
#   - SAPROT embeddings: input_saprot_only/saprot_650M.h5 or path to SAPROT installation
#
# Output:
#   Results are written to output_ELEN_full_model/
###############################################################################

# Exit if any command fails
set -e

# Activate conda environment
source activate elen_test

# Configuration
INPUT_DIR="input_ELEN_full_model"
OUTPUT_DIR="output_ELEN_full_model"
SAPROT_EMB="input_ELEN/saprot_650M.h5"

# Run inference
python ../elen/inference/run_elen_inference.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --feature_mode "full" \
    --saprot_embeddings_file "$SAPROT_EMB" \
    --overwrite