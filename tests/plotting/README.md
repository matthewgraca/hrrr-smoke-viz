# Plotting Library Documentation

This document provides concise documentation for the plotting functions in `libs/plotting.py`.

## Utility Functions

### rmse
```python
rmse(y_pred: np.ndarray, y_test: np.ndarray) -> float
```
Calculates Root Mean Square Error between predictions and ground truth.

### nrmse
```python
nrmse(y_pred: np.ndarray, y_test: np.ndarray) -> float
```
Calculates Normalized Root Mean Square Error between predictions and ground truth.

## Plotting Functions

### plot_prediction_comparison
```python
plot_prediction_comparison(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    sensor_names: List[str],
    sample_idx: int = 0,
    figsize: Tuple[int, int] = (12, 6)
) -> None
```
Bar chart comparing predicted vs actual values for each sensor at a specific time step.

### plot_scatter_comparison
```python
plot_scatter_comparison(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    figsize: Tuple[int, int] = (10, 8)
) -> None
```
Scatter plot of predicted vs actual values with a perfect prediction line.

### plot_error_by_sensor
```python
plot_error_by_sensor(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    sensor_names: List[str],
    figsize: Tuple[int, int] = (10, 6)
) -> None
```
Bar chart of RMSE errors by sensor location.

### plot_time_series_comparison
```python
plot_time_series_comparison(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    sensor_names: List[str],
    figsize: Tuple[int, int] = (12, 8),
    shift_pred: Optional[int] = None,
    remove_outliers: Optional[Tuple[int, int]] = None
) -> None
```
Time series plots comparing predicted vs actual values for each sensor. Supports:
- `shift_pred`: Number of time steps to shift predictions
- `remove_outliers`: Tuple of (start_idx, end_idx) to set predictions to 0

### plot_input_frames
```python
plot_input_frames(
    X: np.ndarray,
    sample_idx: int = 0,
    n_frames: int = 5,
    n_channels: int = 2,
    channel_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 6)
) -> None
```
Visualizes input frames for a given sample. Shows each channel's frames in a grid.

### print_metrics
```python
print_metrics(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    sensor_names: List[str]
) -> None
```
Prints comprehensive RMSE metrics:
- Overall RMSE
- RMSE by frame
- RMSE by sensor location

## Example Usage

See `plotting_demo.py` for a complete example of using all plotting functions with dummy data.
