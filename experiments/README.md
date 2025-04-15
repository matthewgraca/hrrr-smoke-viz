# Experiments
## Experiment 1 - week of April 7th, 2025
[Link to experiment 1](results/experiment-1/README.md)
### Summary
Three experiments with the goal of examining the effect of HRRR on the model's predicitive power for predicting next-frame AirNow PM2.5 stations.

### Settings
- Nearest neighbor interpolation for AirNow sensors.
- ~0.7 degree square bounding box, with 40x40 dimensions.
- 5 frames per sample, sliding window offset by 1 frame. Roughly 164 samples.
- Basic ConvLSTM model.
- Predicting 6 sensors

### Experiment
1. AirNow sensors as only channel.
2. HRRR and AirNow, both matching frames.
3. HRRR with 5-frame future forecast and AirNow \*

\* For a sample, AirNow uses frames at time [0, 4]; HRRR uses forecasts initialized at time 4, forecasting frames at time [5, 9]

## Experiment 2 - week of April 14th, 2025
[Link to experiment 2](results/experiment-2/README.md)
### Summary
Three experiments with the goal of examining the effect of HRRR on the model's predicitive power for predicting next-frame AirNow PM2.5 stations.

### Settings
- Nearest neighbor interpolation for AirNow sensors.
- ~0.3 degree square bounding box, with 200x200 dimensions.
- 5 frames per sample, sliding window offset by 1 frame. Roughly 164 samples.
- Basic ConvLSTM model.
- Predicting 3 sensors

### Experiment
1. AirNow sensors as only channel.
2. HRRR and AirNow, both matching frames.
3. HRRR with 5-frame future forecast and AirNow
