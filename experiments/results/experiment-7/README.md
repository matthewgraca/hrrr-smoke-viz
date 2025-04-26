# Experiment
## Experiment 7 - week of April 21st, 2025
### Summary
Two experiments with the goal of examining the effect of HRRR on the model's predicitive power for predicting next-frame AirNow PM2.5 stations.
### Settings
- IDW interpolation for AirNow sensors.
- **NEW** Using 3 channels; HRRR COLMD and MASSDEN + AirNow data.
- Batch size 4.
- ~0.3 degree square bounding box, with 200x200 dimensions.
- 5 frames per sample, sliding window offset by 1 frame. 165 samples.
- Basic ConvLSTM model.
- Predicting 3 sensors
### Experiment
1. HRRR-COLMD and HRRR-MASSDEN and AirNow, both matching frames.
2. HRRR-COLMD and HRRR-MASSDEN with 5-frame future forecast and AirNow
### Results

1. HRRR+Airnow
```
RESULTS
---------------------------------------------------------------------------
All Days All Locations - y_pred vs y_test Raw RMSE: 15.97
All Days All Locations - y_pred vs y_test RMSE Percent Error of Mean: 144.90%

RESULTS BY FRAME
---------------------------------------------------------------------------
Frame 1 (Hour 1) All Locations - Raw RMSE: 2.36
Frame 1 (Hour 1) All Locations - RMSE Percent Error of Mean: 15.31%

Frame 2 (Hour 2) All Locations - Raw RMSE: 1.70
Frame 2 (Hour 2) All Locations - RMSE Percent Error of Mean: 12.32%

Frame 3 (Hour 3) All Locations - Raw RMSE: 3.47
Frame 3 (Hour 3) All Locations - RMSE Percent Error of Mean: 25.22%

Frame 4 (Hour 4) All Locations - Raw RMSE: 1.37
Frame 4 (Hour 4) All Locations - RMSE Percent Error of Mean: 9.65%

Frame 5 (Hour 5) All Locations - Raw RMSE: 2.80
Frame 5 (Hour 5) All Locations - RMSE Percent Error of Mean: 20.36%

RESULTS BY SENSOR LOCATION
---------------------------------------------------------------------------
All Days - North Holywood Raw RMSE: 5.38
All Days - North Holywood RMSE Percent Error of Mean: 51.50%

All Days - Los Angeles - N. Main Street Raw RMSE: 11.30
All Days - Los Angeles - N. Main Street RMSE Percent Error of Mean: 117.51%

All Days - Compton Raw RMSE: 24.66
All Days - Compton RMSE Percent Error of Mean: 189.91%
```
#### Actual predictions
![](exp_07_a_01.png)
#### Actual predictions left-shifted by 1
![](exp_07_a_02.png)
#### Predictions with outliers removed
![](exp_07_a_03.png)
#### Predictions with outliers removed and left-shifted by 1
![](exp_07_a_04.png)


2. HRRR+Airnow (5-frame forecast)
```
RESULTS
---------------------------------------------------------------------------
All Days All Locations - y_pred vs y_test Raw RMSE: 3.81
All Days All Locations - y_pred vs y_test RMSE Percent Error of Mean: 34.54%

RESULTS BY FRAME
---------------------------------------------------------------------------
Frame 1 (Hour 1) All Locations - Raw RMSE: 2.63
Frame 1 (Hour 1) All Locations - RMSE Percent Error of Mean: 17.10%

Frame 2 (Hour 2) All Locations - Raw RMSE: 1.88
Frame 2 (Hour 2) All Locations - RMSE Percent Error of Mean: 13.69%

Frame 3 (Hour 3) All Locations - Raw RMSE: 3.96
Frame 3 (Hour 3) All Locations - RMSE Percent Error of Mean: 28.73%

Frame 4 (Hour 4) All Locations - Raw RMSE: 1.04
Frame 4 (Hour 4) All Locations - RMSE Percent Error of Mean: 7.32%

Frame 5 (Hour 5) All Locations - Raw RMSE: 2.46
Frame 5 (Hour 5) All Locations - RMSE Percent Error of Mean: 17.92%

RESULTS BY SENSOR LOCATION
---------------------------------------------------------------------------
All Days - North Holywood Raw RMSE: 4.53
All Days - North Holywood RMSE Percent Error of Mean: 43.39%

All Days - Los Angeles - N. Main Street Raw RMSE: 2.41
All Days - Los Angeles - N. Main Street RMSE Percent Error of Mean: 25.01%

All Days - Compton Raw RMSE: 4.13
All Days - Compton RMSE Percent Error of Mean: 31.85%
```
#### Actual predictions
![](exp_07_b_01.png)
#### Actual predictions left-shifted by 1
![](exp_07_b_02.png)
#### Predictions with outliers removed
![](exp_07_b_03.png)
#### Predictions with outliers removed and left-shifted by 1
![](exp_07_b_04.png)

### Thoughts
- Performance is improving, but we still need to look into this offset issue.
