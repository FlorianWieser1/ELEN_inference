#!/bin/bash
#SBATCH -J ELEN_geom_only
#SBATCH -o logs/ELEN_geom_only_%j.log
#SBATCH -e logs/ELEN_geom_only_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

###############################################################################
# ELEN Example Script - Geometry-Only Mode
# Author: Florian Wieser
# Date: 2025-07-29
#
# Description:
#   Runs ELEN in "geom_only" feature mode on example input data.
#   This mode uses only geometric (structure-based) features,
#   without SaProt (sequence/LLM) or Rosetta features.
#   Evaluates using loop-centric pockets (--pocket_type LP).
#
# Usage:
#   sbatch run_elen_geometry_only_model.sh
#
# Requirements:
#   - SLURM cluster with GPU support
#   - Conda environment 'elen_inference' activated, with all ELEN dependencies installed
#   - Input directory: input_ELEN_geometry_only_model
#   - ELEN model weights directory: ../models/
#
# Output:
#   Results are written to output_ELEN_geometry_only_model/
###############################################################################

# Activate conda environment
source activate elen_test

# Configuration
INPUT_DIR="input_ELEN_geometry_only_model"
OUTPUT_DIR="output_ELEN_geometry_only_model"
MODEL_DIR="../models"

# Run ELEN inference (geometry-only features, loop-centric)
python ../elen/inference/run_elen_inference.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --pocket_type "LP" \
    --elen_models_dir "$MODEL_DIR" \
    --feature_mode "geom_only" \
    --overwrite