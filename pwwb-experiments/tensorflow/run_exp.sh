#!/bin/bash

set -e

echo "=== Air Quality Channel Investigation Experiment ==="
echo "Investigating individual channel contributions to model performance"
echo "Starting experiments at $(date)"

RUN_DIR="experiment_output/channel_investigation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo "All experiments will be saved to: $RUN_DIR"

SUMMARY_FILE="$RUN_DIR/experiment_summary.csv"
echo "experiment_id,channels,channel_names,architecture,final_val_loss,training_time,status" > "$SUMMARY_FILE"

echo "Summary will be saved to: $SUMMARY_FILE"

run_experiment() {
    local channels="$1"
    local exp_name="$2"
    local architecture="$3"
    local exp_id="${exp_name}_${architecture}"
    
    echo ""
    echo ">>> Running experiment: $exp_id"
    echo ">>> Channels: $channels" 
    echo ">>> Architecture: $architecture"
    echo ">>> Will save to: $RUN_DIR/$exp_id"
    
    if python3 channel_investigation.py \
        --channels $channels \
        --architecture "$architecture" \
        --experiment-id "$exp_id" \
        --run-dir "$RUN_DIR"; then
        
        if [ -f "$RUN_DIR/$exp_id/results/metadata.json" ]; then
            val_loss=$(python3 -c "import json; data=json.load(open('$RUN_DIR/$exp_id/results/metadata.json')); print(data['final_validation_loss'])")
            training_time=$(python3 -c "import json; data=json.load(open('$RUN_DIR/$exp_id/results/metadata.json')); print(data['training_time_seconds'])")
            channel_names=$(python3 -c "import json; data=json.load(open('$RUN_DIR/$exp_id/results/metadata.json')); print('|'.join(data['channel_names']))")
            
            echo "$exp_id,$channels,$channel_names,$architecture,$val_loss,$training_time,SUCCESS" >> "$SUMMARY_FILE"
            echo "✓ Experiment $exp_id completed successfully! Val Loss: $val_loss"
        else
            echo "$exp_id,$channels,,$architecture,,,FAILED_NO_METADATA" >> "$SUMMARY_FILE"
            echo "✗ Experiment $exp_id completed but no metadata found"
            echo "  Expected metadata at: $RUN_DIR/$exp_id/results/metadata.json"
        fi
    else
        echo "$exp_id,$channels,,$architecture,,,FAILED_TRAINING" >> "$SUMMARY_FILE"
        echo "✗ Experiment $exp_id failed during training"
    fi
}

echo ""
echo "=== Starting Channel Combination Experiments ==="

run_experiment "4" "airnow_only" "classic"
run_experiment "4" "airnow_only" "current"

run_experiment "4 1" "airnow_no2" "classic"
run_experiment "4 1" "airnow_no2" "current"

run_experiment "4 0" "airnow_aod" "classic"
run_experiment "4 0" "airnow_aod" "current"

run_experiment "4 2 3" "airnow_wind" "classic"
run_experiment "4 2 3" "airnow_wind" "current"

run_experiment "4 5" "airnow_hrrr" "classic"
run_experiment "4 5" "airnow_hrrr" "current"

run_experiment "4 1 0" "airnow_no2_aod" "classic"
run_experiment "4 1 0" "airnow_no2_aod" "current"

run_experiment "4 1 2 3" "airnow_no2_wind" "classic"
run_experiment "4 1 2 3" "airnow_no2_wind" "current"

run_experiment "4 1 5" "airnow_no2_hrrr" "classic"
run_experiment "4 1 5" "airnow_no2_hrrr" "current"

run_experiment "4 0 2 3" "airnow_aod_wind" "classic"
run_experiment "4 0 2 3" "airnow_aod_wind" "current"

run_experiment "4 0 5" "airnow_aod_hrrr" "classic"
run_experiment "4 0 5" "airnow_aod_hrrr" "current"

run_experiment "4 2 3 5" "airnow_wind_hrrr" "classic"
run_experiment "4 2 3 5" "airnow_wind_hrrr" "current"

run_experiment "4 1 0 5" "airnow_no2_aod_hrrr" "classic"
run_experiment "4 1 0 5" "airnow_no2_aod_hrrr" "current"

run_experiment "4 1 2 3 5" "airnow_no2_wind_hrrr" "classic"
run_experiment "4 1 2 3 5" "airnow_no2_wind_hrrr" "current"

run_experiment "4 0 2 3 5" "airnow_aod_wind_hrrr" "classic"
run_experiment "4 0 2 3 5" "airnow_aod_wind_hrrr" "current"

run_experiment "4 1 0 2 3" "airnow_no2_aod_wind" "classic"
run_experiment "4 1 0 2 3" "airnow_no2_aod_wind" "current"

run_experiment "4 1 0 2 3 5" "airnow_all_channels" "classic"
run_experiment "4 1 0 2 3 5" "airnow_all_channels" "current"

echo ""
echo "=== All Experiments Completed ==="
echo "Finished at $(date)"
echo "All results saved to: $RUN_DIR"
echo "Summary saved to: $SUMMARY_FILE"

echo ""
echo "=== QUICK RESULTS SUMMARY ==="
echo "Top 5 Best Performing Experiments (by validation loss):"
tail -n +2 "$SUMMARY_FILE" | sort -t',' -k5 -n | head -5 | while IFS=',' read -r exp_id channels channel_names arch val_loss time status; do
    if [ "$status" = "SUCCESS" ]; then
        echo "  $exp_id ($arch): $val_loss | Channels: $channel_names"
    fi
done

echo ""
echo "Failed Experiments:"
grep "FAILED" "$SUMMARY_FILE" | cut -d',' -f1 | while read exp; do
    echo "  $exp"
done

echo ""
echo "Experiment directory: $RUN_DIR"
echo "Individual experiment results in subdirectories of: $RUN_DIR"