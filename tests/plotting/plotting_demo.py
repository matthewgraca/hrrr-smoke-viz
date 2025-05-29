#!/usr/bin/env python3
"""
Plotting Library Demo

This script demonstrates the usage of the plotting library with dummy data.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import plotting module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from libs.plotting import (
    plot_prediction_comparison,
    plot_scatter_comparison,
    plot_error_by_sensor,
    plot_time_series_comparison,
    plot_input_frames,
    print_metrics
)

def generate_dummy_data():
    """Generate dummy data for demonstration purposes."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate dummy sensor names
    sensor_names = ["North Holywood", "Los Angeles - N. Main Street", "Compton"]
    n_sensors = len(sensor_names)

    # Generate dummy time series data
    n_time_steps = 24  # 24 hours
    n_frames = 5  # 5 frames per sample
    n_channels = 2  # HRRR and AirNow channels

    # Generate ground truth data with some trend and noise
    y_test = np.zeros((n_time_steps, n_sensors))
    for i in range(n_sensors):
        trend = np.linspace(10, 30, n_time_steps)
        noise = np.random.normal(0, 5, n_time_steps)
        y_test[:, i] = trend + noise

    # Generate predictions with some error
    y_pred = y_test.copy()
    for i in range(n_sensors):
        error = np.random.normal(0, 3, n_time_steps)
        y_pred[:, i] += error

    # Generate dummy input frames
    X = np.random.rand(1, n_frames, 200, 200, n_channels)  # Single sample with 5 frames

    return X, y_pred, y_test, sensor_names

def main():
    """Main function to demonstrate plotting library."""
    print("Generating dummy data...")
    X, y_pred, y_test, sensor_names = generate_dummy_data()
    
    print("\n1. Plotting prediction comparison...")
    plot_prediction_comparison(y_pred, y_test, sensor_names, sample_idx=12)
    
    print("\n2. Plotting scatter comparison...")
    plot_scatter_comparison(y_pred, y_test)
    
    print("\n3. Plotting error by sensor...")
    plot_error_by_sensor(y_pred, y_test, sensor_names)
    
    print("\n4. Plotting time series comparison...")
    plot_time_series_comparison(y_pred, y_test, sensor_names)
    
    print("\n5. Plotting time series with shifted predictions...")
    plot_time_series_comparison(y_pred, y_test, sensor_names, shift_pred=1)
    
    print("\n6. Plotting time series with outliers removed...")
    plot_time_series_comparison(y_pred, y_test, sensor_names, remove_outliers=(10, 15))
    
    print("\n7. Plotting input frames...")
    plot_input_frames(X, channel_names=["HRRR", "AirNow"])
    
    print("\n8. Printing metrics...")
    print_metrics(y_pred, y_test, sensor_names)

if __name__ == "__main__":
    main() 