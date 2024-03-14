#!/bin/bash

# Define base command
BASE_CMD="python main.py"

# Example experiments
# Experiment 1: CNN with varying width
for width in 2 4 8 16 32; do
    $BASE_CMD --model CNN --width $width --epochs 250 --lr 0.01 --decay 1e-4 
done

# Experiment 2: ResNet with varying depth
for depth in 8 16 20 32; do
    $BASE_CMD --model ResNet18 --depth $depth --epochs 250 --lr 0.01 --decay 1e-3
done

# Experiment 3: Different decay rates
for decay in 1e2 1e-1 1e-2 1e-3 1e-4 1e-6; do
    $BASE_CMD --model ResNet18 --width 16 --depth 20 --epochs 250 --lr 0.01 --decay $decay
done
