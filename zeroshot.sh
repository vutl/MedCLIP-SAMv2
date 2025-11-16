#!/bin/bash

# custom config

# Enter the path to your dataset
DATASET=$1

# ============================================================================
# Zero-Shot Segmentation Pipeline
# ============================================================================
# This script runs the complete MedCLIP-SAMv2 zero-shot pipeline in 3 stages:
#
# Stage 1: Generate Saliency Maps (~46% DSC if evaluated)
#   - Uses fine-tuned BiomedCLIP model and M2IB to generate raw attribution maps
#   - Output: saliency_map_outputs/${DATASET}/masks
#
# Stage 2: Post-process Saliency Maps (~58% DSC if evaluated)
#   - Binarizes saliency maps using k-means clustering
#   - Creates "coarse" segmentation masks
#   - Output: coarse_outputs/${DATASET}/masks
#
# Stage 3: SAM Refinement (~78% DSC - FINAL OUTPUT)
#   - Uses coarse masks to generate bounding box prompts for SAM
#   - Produces final, refined segmentation masks
#   - Output: sam_outputs/${DATASET}/masks  <-- EVALUATE THIS DIRECTORY
#
# IMPORTANT: When evaluating results, use sam_outputs/${DATASET}/masks
# ============================================================================

# Stage 1: Generate Saliency Maps
# NOTE: --hyper-opt flag has been removed. It uses a fragile 3-sample optimization
# that is non-deterministic and can lead to poor results. The script now uses
# stable default parameters (vbeta=0.1, vvar=1.0, vlayer=7) as reported in the paper.
python saliency_maps/generate_saliency_maps.py \
--input-path ${DATASET}/images \
--output-path saliency_map_outputs/${DATASET}/masks \
--val-path ${DATASET}/val_images \
--model-name BiomedCLIP \
--finetuned

# Stage 2: Post-process Saliency Maps
python postprocessing/postprocess_saliency_maps.py \
--input-path ${DATASET}/images \
--output-path coarse_outputs/${DATASET}/masks \
--sal-path saliency_map_outputs/${DATASET}/masks \
--postprocess kmeans \
--filter
# --num-contours 2 # number of contours to extract, for lungs, use 2 contours

# Stage 3: SAM Refinement (FINAL OUTPUT - Evaluate this directory!)
python segment-anything/prompt_sam.py \
--input ${DATASET}/images \
--mask-input coarse_outputs/${DATASET}/masks \
--output sam_outputs/${DATASET}/masks \
--model-type vit_h \
--checkpoint segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth \
--prompts boxes \
# --multicontour # for lungs, use this flag
