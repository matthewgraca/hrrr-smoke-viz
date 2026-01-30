#!/bin/bash

# -- PARAMETERS --
# model name = name of the model you want to train
# loss name = name of the loss you want to use
# experiment name = string to be append to experiment description
# data location = folder containing the training data (in the current directory)
# is test = is/is not a test. turn false if you want a real run.
# viz only = run only the visualization script, no training.

MODEL_NAME='dual_autoencoder'
LOSS_NAME='grid_mae'
EXPERIMENT_NAME='84x84'
DATA_LOC='/mnt/wildfire/training-data/2026-01-27'
IS_TEST=true
VIZ_ONLY=false

if [ "$VIZ_ONLY" = true ]; then
    echo -e "\e[1mConducting visualizations only.\e[0m"
    python vis.py \
        "${MODEL_NAME}_${LOSS_NAME}_loss_${EXPERIMENT_NAME}" \
        "$DATA_LOC"
    exit 0
fi

if [ "$IS_TEST" = true ]; then
    echo -e "\e[1mConducting test experiment only.\e[0m"
    python training.py \
        "$MODEL_NAME" \
        "$LOSS_NAME" \
        "$DATA_LOC" \
        -r "$EXPERIMENT_NAME" \
        -t
else
    echo -e "\e[1mRunning full training sequence.\e[0m"
    python training.py \
        "$MODEL_NAME" \
        "$LOSS_NAME" \
        "$DATA_LOC" \
        -r "$EXPERIMENT_NAME"
    python vis.py \
        "${MODEL_NAME}_${LOSS_NAME}_loss_${EXPERIMENT_NAME}" \
        "$DATA_LOC"
fi

