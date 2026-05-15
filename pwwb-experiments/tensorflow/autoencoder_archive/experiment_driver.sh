#!/bin/bash

# -- PARAMETERS -> see settings.conf --
# -- Assumptions --
# Script assumes that training.py is in the current directory.
# settings.conf is in the same directory that defines all the args

if [[ -f "settings.conf" ]]; then
    source settings.conf
    echo -e "\e[1mLoading config.\e[0m"
    grep "=" settings.conf
    echo ""
fi

if [ "$VIZ_ONLY" = true ]; then
    echo -e "\e[1mConducting visualizations only.\e[0m"
    if [ "$MINMAX_TARGET" = true ]; then
        python /home/mgraca/Workspace/hrrr-smoke-viz/visualizations/prediction_viz_and_metrics.py \
            --preds "${EXPERIMENT_LOC}/y_pred.npy" \
            --trues "${DATA_LOC}/Y_valid.npy" \
            --inputs "${DATA_LOC}/X_valid.npy" \
            --out-dir "${EXPERIMENT_LOC}" \
            --fire-name "${FIRE_NAME}" \
            --sensor-locations "${RAW_FIRE_LOC}/airnow_processed.npz" \
            --training-history "${EXPERIMENT_LOC}/history.pkl" \
            --target-scaler "${DATA_LOC}/target_scaler.pkl"
    else
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
fi

if [ "$IS_TEST" = true ]; then
    echo -e "\e[1mConducting test experiment only (forward pass).\e[0m"
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
    if [ "$MINMAX_TARGET" = true ]; then
        python /home/mgraca/Workspace/hrrr-smoke-viz/visualizations/prediction_viz_and_metrics.py \
            --preds "${EXPERIMENT_LOC}/y_pred.npy" \
            --trues "${DATA_LOC}/Y_valid.npy" \
            --inputs "${DATA_LOC}/X_valid.npy" \
            --out-dir "${EXPERIMENT_LOC}" \
            --fire-name "${FIRE_NAME}" \
            --sensor-locations "${RAW_FIRE_LOC}/airnow_processed.npz" \
            --training-history "${EXPERIMENT_LOC}/history.pkl" \
            --target-scaler "${DATA_LOC}/target_scaler.pkl"

    else
        python /home/mgraca/Workspace/hrrr-smoke-viz/visualizations/prediction_viz_and_metrics.py \
            --preds "${EXPERIMENT_LOC}/y_pred.npy" \
            --trues "${DATA_LOC}/Y_valid.npy" \
            --inputs "${DATA_LOC}/X_valid.npy" \
            --out-dir "${EXPERIMENT_LOC}" \
            --fire-name "${FIRE_NAME}" \
            --sensor-locations "${RAW_FIRE_LOC}/airnow_processed.npz" \
            --training-history "${EXPERIMENT_LOC}/history.pkl"
    fi
fi

