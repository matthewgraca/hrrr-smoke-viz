import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Tuple
from skimage.metrics import mean_squared_error

def rmse(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    """Calculate Root Mean Square Error between predictions and ground truth."""
    return np.sqrt(mean_squared_error(y_pred, y_test))

def nrmse(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    """Calculate Normalized Root Mean Square Error between predictions and ground truth."""
    return rmse(y_pred, y_test) / np.mean(y_test) * 100

def plot_prediction_comparison(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    sensor_names: List[str],
    sample_idx: int = 0,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Create a bar chart comparing predicted vs actual values for each sensor.
    
    Args:
        y_pred: Array of predicted values
        y_test: Array of ground truth values
        sensor_names: List of sensor location names
        sample_idx: Index of sample to plot
        figsize: Figure size as (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(sensor_names))
    width = 0.35

    true_vals = y_test[sample_idx]
    pred_vals = y_pred[sample_idx]

    rects1 = ax.bar(x - width/2, true_vals, width, label='Actual')
    rects2 = ax.bar(x + width/2, pred_vals, width, label='Predicted')

    ax.set_title('PM2.5 Actual vs. Predicted Values by Sensor Location')
    ax.set_ylabel('PM2.5 Value')
    ax.set_xlabel('Sensor Location')
    ax.set_xticks(x)
    ax.set_xticklabels(sensor_names, rotation=45, ha='right')
    ax.legend()
    plt.show()

def plot_scatter_comparison(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Create a scatter plot of predicted vs actual values.
    
    Args:
        y_pred: Array of predicted values
        y_test: Array of ground truth values
        figsize: Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    plt.scatter(y_test.flatten(), y_pred.flatten(), alpha=0.5)
    plt.plot([0, np.max(y_test)], [0, np.max(y_test)], 'r--')  # Perfect prediction line
    plt.xlabel('Actual PM2.5 Values')
    plt.ylabel('Predicted PM2.5 Values')
    plt.title('Actual vs. Predicted PM2.5 Values')
    plt.grid(True)
    plt.show()

def plot_error_by_sensor(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    sensor_names: List[str],
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Create a bar chart of RMSE errors by sensor location.
    
    Args:
        y_pred: Array of predicted values
        y_test: Array of ground truth values
        sensor_names: List of sensor location names
        figsize: Figure size as (width, height)
    """
    error_by_sensor = []
    for i in range(len(sensor_names)):
        error = rmse(y_pred[:, i], y_test[:, i])
        error_by_sensor.append(error)

    plt.figure(figsize=figsize)
    plt.bar(sensor_names, error_by_sensor)
    plt.ylabel('RMSE')
    plt.title('Prediction Error by Sensor Location')
    plt.xticks(rotation=45, ha='right')
    plt.show()

def plot_time_series_comparison(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    sensor_names: List[str],
    figsize: Tuple[int, int] = (16, 24),  # Increased height
    shift_pred: Optional[int] = None,
    remove_outliers: Optional[Tuple[int, int]] = None,
    subplot_height: float = 2.0  # Height per subplot in inches
) -> None:
    """
    Create time series plots comparing predicted vs actual values for each sensor.
    
    Args:
        y_pred: Array of predicted values
        y_test: Array of ground truth values
        sensor_names: List of sensor location names
        figsize: Figure size as (width, height)
        shift_pred: Number of time steps to shift predictions (e.g. 1 for left shift)
        remove_outliers: Tuple of (start_idx, end_idx) to set predictions to 0 in that range
        subplot_height: Height per subplot in inches for better spacing
    """
    # Calculate dynamic figure height based on number of sensors and desired subplot height
    num_sensors = len(sensor_names)
    fig_height = num_sensors * subplot_height
    plt.figure(figsize=(figsize[0], fig_height))
    
    # Handle optional modifications to predictions
    y_pred_plot = y_pred.copy()
    if remove_outliers is not None:
        start_idx, end_idx = remove_outliers
        y_pred_plot[start_idx:end_idx] = 0.0
    if shift_pred is not None:
        y_pred_plot = y_pred_plot[shift_pred:]
        y_test_plot = y_test[:-shift_pred] if shift_pred > 0 else y_test.copy()
    else:
        y_test_plot = y_test.copy()

    for i, sensor in enumerate(sensor_names):
        ax = plt.subplot(num_sensors, 1, i + 1)
        
        # Plot data with distinct colors and markers for better visibility
        plt.plot(y_test_plot[:, i], label='Actual', marker='o', markersize=3, 
                 color='#1f77b4', linewidth=1.5, markevery=5)
        plt.plot(y_pred_plot[:, i], label='Predicted', marker='x', markersize=4,
                 color='#ff7f0e', linewidth=1.5, markevery=5)
        
        # Add grid with light color for better readability
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Improve title and labels
        plt.title(f'Sensor: {sensor}', fontsize=12, fontweight='bold')
        plt.ylabel('PM2.5', fontsize=10)
        
        # Only add x-label for the bottom subplot
        if i == num_sensors - 1:
            plt.xlabel('Time Step', fontsize=10)
        
        # Add legend for each subplot with better placement
        plt.legend(loc='upper right')
        
        # Set y limits with a bit of padding for better visualization
        y_max = max(np.max(y_test_plot[:, i]), np.max(y_pred_plot[:, i]))
        y_min = min(np.min(y_test_plot[:, i]), np.min(y_pred_plot[:, i]))
        padding = (y_max - y_min) * 0.1
        plt.ylim(y_min - padding, y_max + padding)
    
    plt.tight_layout(pad=2.0)  # Increase padding between subplots
    plt.subplots_adjust(hspace=0.5)  # Add more space between subplots
    
    # Add overall title
    title_text = 'PM2.5 Actual vs. Predicted Values by Sensor Location'
    if shift_pred is not None:
        title_text += f' (Predictions Shifted by {shift_pred} Steps)'
    plt.suptitle(title_text, fontsize=16, y=0.995)
    
    plt.show()

def plot_input_frames(
    X: np.ndarray,
    sample_idx: int = 0,
    n_frames: int = 5,
    n_channels: int = 2,
    channel_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 6)
) -> None:
    """
    Plot input frames for a given sample.
    
    Args:
        X: Input data array
        sample_idx: Index of sample to plot
        n_frames: Number of frames to plot
        n_channels: Number of channels in the data
        channel_names: Names of the channels
        figsize: Figure size as (width, height)
    """
    if channel_names is None:
        channel_names = [f'Channel {i}' for i in range(n_channels)]
    
    fig, axes = plt.subplots(n_channels, n_frames, figsize=figsize)
    
    for channel in range(n_channels):
        for frame in range(n_frames):
            ax = axes[channel, frame]
            im = ax.imshow(np.squeeze(X[sample_idx, frame, :, :, channel]), cmap='viridis')
            ax.set_title(f"{channel_names[channel]} Frame {frame + 1}")
            ax.axis("off")
    
    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    plt.suptitle(f'Input data for sample {sample_idx}')
    plt.show()

def print_metrics(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    sensor_names: List[str]
) -> None:
    """
    Print RMSE metrics for predictions.
    
    Args:
        y_pred: Array of predicted values
        y_test: Array of ground truth values
        sensor_names: List of sensor location names
    """
    print("RESULTS")
    print("---------------------------------------------------------------------------")
    print(f"All Days All Locations - y_pred vs y_test Raw RMSE: {rmse(y_pred, y_test):.2f}")
    print(f"All Days All Locations - y_pred vs y_test RMSE Percent Error of Mean: {nrmse(y_pred, y_test):.2f}%\n")

    print("RESULTS BY FRAME")
    print("---------------------------------------------------------------------------")
    for i in range(y_pred.shape[0]):
        print(f"Frame {i+1} (Hour {i+1}) All Locations - Raw RMSE: {rmse(y_pred[i,:], y_test[i,:]):.2f}")
        print(f"Frame {i+1} (Hour {i+1}) All Locations - RMSE Percent Error of Mean: {nrmse(y_pred[i,:], y_test[i,:]):.2f}%\n")

    print("RESULTS BY SENSOR LOCATION")
    print("---------------------------------------------------------------------------")
    for i, loc in enumerate(sensor_names):
        print(f"All Days - {loc} Raw RMSE: {rmse(y_pred[:,i], y_test[:,i]):.2f}")
        print(f"All Days - {loc} RMSE Percent Error of Mean: {nrmse(y_pred[:,i], y_test[:,i]):.2f}%\n") 