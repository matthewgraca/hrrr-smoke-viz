#!/bin/bash

# -- PARAMETERS --
# model name = name of the model you want to train
# loss name = name of the loss you want to use
# experiment name = string to be append to experiment description
# data location = folder containing the training data (in the current directory)
# is test = is/is not a test. turn false if you want a real run.

MODEL_NAME='dual_autoencoder'
LOSS_NAME='grid_mse'
EXPERIMENT_NAME='0'
DATA_LOC='preprocessed_cache'
IS_TEST=false

if [ "$IS_TEST" = true ]; then
    python training.py \
        "$MODEL_NAME" \
        "$LOSS_NAME" \
        "$DATA_LOC" \
        -r "$EXPERIMENT_NAME" \
        -t
    python vis.py \
        "test_experiment" \
        "$DATA_LOC"
else
    python training.py \
        "$MODEL_NAME" \
        "$LOSS_NAME" \
        "$DATA_LOC" \
        -r "$EXPERIMENT_NAME"
    python vis.py \
        "${MODEL_NAME}_${LOSS_NAME}_loss_${EXPERIMENT_NAME}" \
        "$DATA_LOC"
fi

