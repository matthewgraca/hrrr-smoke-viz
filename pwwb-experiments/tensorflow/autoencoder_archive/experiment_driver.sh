#!/bin/bash

# -- PARAMETERS --
# model name = name of the model you want to train
# loss name = name of the loss you want to use
# epochs = number of epochs to train for
# experiment name = string to be append to experiment description
# data location = folder containing the training data (in the current directory)
# is test = is/is not a test. turn false if you want a real run.
# viz only = run only the visualization script, no training.

# -- Assumptions --
# Script assumes that training.py is in the current directory, so make sure to
#   run the script where the training and vis script is.

MODEL_NAME='dual_ae_conv2d_bn'
LOSS_NAME='grid_mae'
EPOCHS='100'
BATCH_SIZE='12'
DATA_LOC='/mnt/wildfire/processed-data/2026-04-29/dataset'
RESULTS_LOC='/mnt/wildfire/experiments'
EXPERIMENT_NAME='1'

FIRE_NAME='palisades_eaton'
RAW_FIRE_LOC='/mnt/wildfire/processed-data/2026-04-29/la/palisades_eaton'

EXPERIMENT_LOC="${RESULTS_LOC}/${MODEL_NAME}_${LOSS_NAME}_${EXPERIMENT_NAME}"
IS_TEST=false
VIZ_ONLY=true

if [ "$VIZ_ONLY" = true ]; then
    echo -e "\e[1mConducting visualizations only.\e[0m"
    python /home/mgraca/Workspace/hrrr-smoke-viz/visualizations/prediction_viz_and_metrics.py \
        --preds "${EXPERIMENT_LOC}/y_pred.npy" \
        --trues "${DATA_LOC}/Y_valid.npy" \
        --inputs "${DATA_LOC}/X_valid.npy" \
        --out-dir "${EXPERIMENT_LOC}" \
        --fire-name "${FIRE_NAME}" \
        --sensor-locations "${RAW_FIRE_LOC}/airnow_processed.npz" \
        --training-history "${EXPERIMENT_LOC}/history.pkl"
    exit 0
fi

if [ "$IS_TEST" = true ]; then
    echo -e "\e[1mConducting test experiment only.\e[0m"
    python training.py \
        "$MODEL_NAME" \
        "$LOSS_NAME" \
        "$EPOCHS" \
        "$BATCH_SIZE" \
        "$DATA_LOC" \
        "$RESULTS_LOC" \
        "$EXPERIMENT_NAME" \
        -t
else
    echo -e "\e[1mRunning full training sequence.\e[0m"
    python training.py \
        "$MODEL_NAME" \
        "$LOSS_NAME" \
        "$EPOCHS" \
        "$BATCH_SIZE" \
        "$DATA_LOC" \
        "$RESULTS_LOC" \
        "$EXPERIMENT_NAME"

    python /home/mgraca/Workspace/hrrr-smoke-viz/visualizations/prediction_viz_and_metrics.py \
        --preds "${EXPERIMENT_LOC}/y_pred.npy" \
        --trues "${DATA_LOC}/Y_valid.npy" \
        --inputs "${DATA_LOC}/X_valid.npy" \
        --out-dir "${EXPERIMENT_LOC}" \
        --fire-name "${FIRE_NAME}" \
        --sensor-locations "${RAW_FIRE_LOC}/airnow_processed.npz" \
        --training-history "${EXPERIMENT_LOC}/history.pkl"
fi

