import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from skimage.metrics import mean_squared_error
import seaborn as sns

def rmse(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    """Calculate Root Mean Square Error between predictions and ground truth."""
    return np.sqrt(mean_squared_error(y_pred, y_test))

def mae(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    """Calculate Mean Absolute Error between predictions and ground truth."""
    return np.mean(np.abs(y_pred - y_test))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def nrmse(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    """Calculate Normalized Root Mean Square Error as percentage."""
    return rmse(y_pred, y_test) / np.mean(y_test) * 100

def plot_frame_by_frame_rmse(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot RMSE for each frame showing temporal degradation patterns.
    Essential for ConvLSTM as it reveals how prediction accuracy decreases over time horizons.
    """
    if len(y_pred.shape) != 3:
        print("Frame-by-frame analysis requires multi-frame predictions")
        return
    
    print("📊 TEMPORAL DEGRADATION ANALYSIS")
    print("This measures how ConvLSTM prediction accuracy degrades as we predict further into the future.")
    print("Later time steps are inherently harder to predict due to error accumulation.\n")
    
    num_frames = y_pred.shape[1]
    frame_rmse = []
    frame_nrmse = []
    
    for frame in range(num_frames):
        frame_pred = y_pred[:, frame, :].flatten()
        frame_test = y_test[:, frame, :].flatten()
        
        rmse_val = rmse(frame_pred, frame_test)
        nrmse_val = nrmse(frame_pred, frame_test)
        
        frame_rmse.append(rmse_val)
        frame_nrmse.append(nrmse_val)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # RMSE plot
    frames = range(1, num_frames + 1)
    colors = ['green' if x < 1.5 else 'gold' if x < 3.0 else 'orange' if x < 5.0 else 'red' for x in frame_rmse]
    bars1 = ax1.bar(frames, frame_rmse, alpha=0.8, color=colors, edgecolor='black')
    ax1.set_ylabel('RMSE')
    ax1.set_xlabel('Future Frame (Hour)')
    ax1.set_title('RMSE by Future Frame')
    ax1.grid(True, alpha=0.3)
    
    for bar, rmse_val in zip(bars1, frame_rmse):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{rmse_val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # NRMSE plot
    nrmse_colors = ['green' if x < 15.0 else 'gold' if x < 30.0 else 'orange' if x < 50.0 else 'red' for x in frame_nrmse]
    bars2 = ax2.bar(frames, frame_nrmse, alpha=0.8, color=nrmse_colors, edgecolor='black')
    ax2.set_ylabel('NRMSE (%)')
    ax2.set_xlabel('Future Frame (Hour)')
    ax2.set_title('Normalized RMSE by Future Frame')
    ax2.grid(True, alpha=0.3)
    
    for bar, nrmse_val in zip(bars2, frame_nrmse):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{nrmse_val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\n📊 FRAME-BY-FRAME RMSE SUMMARY:")
    print("=" * 50)
    for frame, (rmse_val, nrmse_val) in enumerate(zip(frame_rmse, frame_nrmse), 1):
        print(f"Hour {frame}: RMSE = {rmse_val:.3f} ({nrmse_val:.1f}%)")

def plot_avg_rmse_per_station(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    sensor_names: List[str],
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """
    Plot average RMSE per station across all time horizons.
    Reveals spatial patterns and identifies challenging locations for ConvLSTM.
    """
    if len(y_pred.shape) != 3:
        print("Station RMSE analysis requires multi-frame predictions")
        return
    
    print("📍 SPATIAL PERFORMANCE ANALYSIS")
    print("This identifies which monitoring stations are consistently difficult for ConvLSTM to predict.")
    print("Challenging stations often have complex local meteorology or unique pollution sources.\n")
    
    num_sensors = len(sensor_names)
    station_rmse = []
    station_nrmse = []
    
    for sensor_idx in range(num_sensors):
        sensor_pred = y_pred[:, :, sensor_idx].flatten()
        sensor_test = y_test[:, :, sensor_idx].flatten()
        
        rmse_val = rmse(sensor_pred, sensor_test)
        nrmse_val = nrmse(sensor_pred, sensor_test)
        
        station_rmse.append(rmse_val)
        station_nrmse.append(nrmse_val)
    
    sorted_indices = np.argsort(station_rmse)
    sorted_stations = [sensor_names[i] for i in sorted_indices]
    sorted_rmse = [station_rmse[i] for i in sorted_indices]
    sorted_nrmse = [station_nrmse[i] for i in sorted_indices]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    colors = ['green' if x < 1.5 else 'gold' if x < 3.0 else 'orange' if x < 5.0 else 'red' for x in sorted_rmse]
    bars1 = ax1.barh(range(num_sensors), sorted_rmse, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Average RMSE (All Hours)')
    ax1.set_ylabel('Sensor Location')
    ax1.set_title('Average RMSE per Station (All Hours)')
    ax1.set_yticks(range(num_sensors))
    ax1.set_yticklabels([name[:20] + '...' if len(name) > 20 else name for name in sorted_stations])
    ax1.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, rmse_val) in enumerate(zip(bars1, sorted_rmse)):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{rmse_val:.2f}', va='center', fontweight='bold')
    
    nrmse_colors = ['green' if x < 15.0 else 'gold' if x < 30.0 else 'orange' if x < 50.0 else 'red' for x in sorted_nrmse]
    bars2 = ax2.barh(range(num_sensors), sorted_nrmse, color=nrmse_colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Average NRMSE (%) (All Hours)')
    ax2.set_ylabel('Sensor Location')
    ax2.set_title('Average NRMSE per Station (All Hours)')
    ax2.set_yticks(range(num_sensors))
    ax2.set_yticklabels([name[:20] + '...' if len(name) > 20 else name for name in sorted_stations])
    ax2.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, nrmse_val) in enumerate(zip(bars2, sorted_nrmse)):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{nrmse_val:.1f}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    excellent = [(name, rmse_val) for name, rmse_val in zip(sorted_stations, sorted_rmse) if rmse_val < 1.5]
    good = [(name, rmse_val) for name, rmse_val in zip(sorted_stations, sorted_rmse) if 1.5 <= rmse_val < 3.0]
    fair = [(name, rmse_val) for name, rmse_val in zip(sorted_stations, sorted_rmse) if 3.0 <= rmse_val < 5.0]
    poor = [(name, rmse_val) for name, rmse_val in zip(sorted_stations, sorted_rmse) if rmse_val >= 5.0]
    
    print("\n📍 STATION PERFORMANCE SUMMARY (All Hours Combined)")
    print("=" * 80)
    
    print(f"\n🟢 EXCELLENT Stations (RMSE < 1.5): {len(excellent)}")
    for name, rmse_val in excellent:
        print(f"   • {name}: {rmse_val:.3f}")
    
    print(f"\n🟡 GOOD Stations (1.5-3.0): {len(good)}")
    for name, rmse_val in good:
        print(f"   • {name}: {rmse_val:.3f}")
    
    print(f"\n🟠 FAIR Stations (3.0-5.0): {len(fair)}")
    for name, rmse_val in fair:
        print(f"   • {name}: {rmse_val:.3f}")
    
    print(f"\n🔴 POOR Stations (≥ 5.0): {len(poor)}")
    for name, rmse_val in poor:
        print(f"   • {name}: {rmse_val:.3f}")
    
    best_station = sorted_stations[0]
    worst_station = sorted_stations[-1]
    avg_rmse = np.mean(sorted_rmse)
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   🏆 Best Station:  {best_station} (RMSE: {sorted_rmse[0]:.3f})")
    print(f"   💥 Worst Station: {worst_station} (RMSE: {sorted_rmse[-1]:.3f})")
    print(f"   📈 Average RMSE:  {avg_rmse:.3f}")
    print(f"   📏 RMSE Range:    {sorted_rmse[-1] - sorted_rmse[0]:.3f}")
    print(f"   🎯 Reliability:   {len(excellent + good)}/{len(sensor_names)} stations ≤ 3.0 RMSE")

def print_summary_table(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    sensor_names: List[str]
) -> None:
    """
    Print compact summary table showing spatio-temporal performance patterns.
    """
    if len(y_pred.shape) != 3:
        print("Summary table requires multi-frame predictions")
        return
    
    num_frames = y_pred.shape[1]
    
    print("\n📊 SUMMARY TABLE - RMSE BY HOUR AND SENSOR")
    print("=" * 120)
    
    rmse_matrix = np.zeros((len(sensor_names), num_frames))
    for frame in range(num_frames):
        for sensor_idx in range(len(sensor_names)):
            sensor_pred = y_pred[:, frame, sensor_idx]
            sensor_test = y_test[:, frame, sensor_idx]
            rmse_matrix[sensor_idx, frame] = rmse(sensor_pred, sensor_test)
    
    sensor_width = 25
    hour_width = 8
    avg_width = 8
    
    header = f"{'Sensor':<{sensor_width}}"
    for hour in range(1, num_frames + 1):
        header += f"│{'Hour ' + str(hour):^{hour_width}}"
    header += f"│{'Avg':^{avg_width}}"
    print(header)
    
    separator = "─" * sensor_width
    for _ in range(num_frames):
        separator += "┼" + "─" * hour_width
    separator += "┼" + "─" * avg_width
    print(separator)
    
    for sensor_idx, sensor in enumerate(sensor_names):
        display_name = sensor[:sensor_width-1] if len(sensor) > sensor_width-1 else sensor
        row = f"{display_name:<{sensor_width}}"
        
        sensor_avg = np.mean(rmse_matrix[sensor_idx, :])
        
        for frame in range(num_frames):
            rmse_val = rmse_matrix[sensor_idx, frame]
            row += f"│{rmse_val:^{hour_width}.2f}"
        
        row += f"│{sensor_avg:^{avg_width}.2f}"
        print(row)
    
    print(separator)
    footer = f"{'HOURLY AVERAGE':<{sensor_width}}"
    overall_avg = np.mean(rmse_matrix)
    
    for frame in range(num_frames):
        hour_avg = np.mean(rmse_matrix[:, frame])
        footer += f"│{hour_avg:^{hour_width}.2f}"
    footer += f"│{overall_avg:^{avg_width}.2f}"
    print(footer)
    
    print(f"\n🎯 KEY: Excellent(<1.5) Good(1.5-3.0) Fair(3.0-5.0) Poor(≥5.0)")

def plot_frame_time_series(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    sensor_names: List[str],
    frame_idx: int = 0,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """
    Plot time series for a specific frame across all sensors.
    Shows ConvLSTM's ability to capture temporal patterns at each prediction horizon.
    """
    if len(y_pred.shape) != 3:
        print("Time series analysis requires multi-frame predictions")
        return
    
    frame_pred = y_pred[:, frame_idx, :]
    frame_test = y_test[:, frame_idx, :]
    
    num_sensors = len(sensor_names)
    num_cols = 2
    num_rows = (num_sensors + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, sensor in enumerate(sensor_names):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        
        ax.plot(frame_test[:, i], label='Actual', alpha=0.8, linewidth=2)
        ax.plot(frame_pred[:, i], label='Predicted', alpha=0.8, linewidth=2)
        
        sensor_rmse = rmse(frame_pred[:, i], frame_test[:, i])
        sensor_r2 = r2_score(frame_test[:, i], frame_pred[:, i])
        
        ax.set_title(f'{sensor}\nRMSE: {sensor_rmse:.2f}, R²: {sensor_r2:.3f}')
        ax.set_xlabel('Sample')
        ax.set_ylabel('PM2.5 (μg/m³)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    for i in range(num_sensors, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle(f'Time Series Analysis - Frame {frame_idx + 1} (Hour {frame_idx + 1})', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()

def plot_frame_scatter(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    frame_idx: int = 0,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot scatter plot for a specific frame showing prediction accuracy.
    Critical for ConvLSTM evaluation as it reveals prediction bias and variance patterns.
    """
    if len(y_pred.shape) != 3:
        print("Scatter analysis requires multi-frame predictions")
        return
    
    frame_pred = y_pred[:, frame_idx, :].flatten()
    frame_test = y_test[:, frame_idx, :].flatten()
    
    plt.figure(figsize=figsize)
    plt.scatter(frame_test, frame_pred, alpha=0.6, s=20)
    
    min_val = min(np.min(frame_test), np.min(frame_pred))
    max_val = max(np.max(frame_test), np.max(frame_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    rmse_val = rmse(frame_pred, frame_test)
    r2_val = r2_score(frame_test, frame_pred)
    mae_val = mae(frame_pred, frame_test)
    nrmse_val = nrmse(frame_pred, frame_test)
    
    plt.xlabel('Actual PM2.5 (μg/m³)')
    plt.ylabel('Predicted PM2.5 (μg/m³)')
    plt.title(f'Frame {frame_idx + 1} (Hour {frame_idx + 1}) - Actual vs Predicted\n'
              f'RMSE: {rmse_val:.2f} ({nrmse_val:.1f}%), MAE: {mae_val:.2f}, R²: {r2_val:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_frame_heatmap(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    sensor_names: List[str],
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """
    Plot heatmap showing RMSE patterns across space and time.
    Reveals spatio-temporal error patterns critical for ConvLSTM model improvement.
    """
    if len(y_pred.shape) != 3:
        print("Heatmap analysis requires multi-frame predictions")
        return
    
    print("🔥 SPATIO-TEMPORAL ERROR ANALYSIS")
    print("This heatmap reveals how ConvLSTM errors vary across both space (stations) and time (hours).")
    print("Dark red areas indicate challenging spatio-temporal combinations requiring model attention.\n")
    
    num_frames = y_pred.shape[1]
    num_sensors = len(sensor_names)
    
    rmse_matrix = np.zeros((num_sensors, num_frames))
    
    for frame in range(num_frames):
        for sensor in range(num_sensors):
            sensor_pred = y_pred[:, frame, sensor]
            sensor_test = y_test[:, frame, sensor]
            rmse_matrix[sensor, frame] = rmse(sensor_pred, sensor_test)
    
    plt.figure(figsize=figsize)
    
    sns.heatmap(rmse_matrix, 
                xticklabels=[f'Hour {i+1}' for i in range(num_frames)],
                yticklabels=sensor_names,
                annot=True, 
                fmt='.2f', 
                cmap='YlOrRd',
                cbar_kws={'label': 'RMSE'})
    
    plt.title('RMSE Heatmap: Sensors vs Future Frames')
    plt.xlabel('Future Frame (Hour)')
    plt.ylabel('Sensor Location')
    plt.tight_layout()
    plt.show()
    
    print("\n🔥 WORST PERFORMING COMBINATIONS:")
    print("=" * 40)
    flat_indices = np.argsort(rmse_matrix.flatten())[-5:]
    for idx in reversed(flat_indices):
        sensor_idx, frame_idx = np.unravel_index(idx, rmse_matrix.shape)
        print(f"{sensor_names[sensor_idx]} at Hour {frame_idx+1}: RMSE = {rmse_matrix[sensor_idx, frame_idx]:.3f}")

def print_detailed_frame_stats(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    sensor_names: List[str]
) -> None:
    """
    Print detailed statistics for every frame and sensor combination.
    Provides comprehensive ConvLSTM performance breakdown for model diagnostics.
    """
    if len(y_pred.shape) != 3:
        print("Detailed stats require multi-frame predictions")
        return
    
    print("📋 DETAILED CONVLSTM PERFORMANCE BREAKDOWN")
    print("This section provides comprehensive metrics for each prediction horizon and monitoring station.")
    print("Essential for identifying specific model weaknesses and guiding improvements.\n")
    
    num_frames = y_pred.shape[1]
    
    print("📋 DETAILED FRAME-BY-FRAME STATISTICS")
    print("=" * 100)
    
    for frame in range(num_frames):
        print(f"\n🕐 HOUR {frame + 1} ANALYSIS")
        print("─" * 100)
        
        frame_pred = y_pred[:, frame, :].flatten()
        frame_test = y_test[:, frame, :].flatten()
        
        overall_rmse = rmse(frame_pred, frame_test)
        overall_mae = mae(frame_pred, frame_test)
        overall_r2 = r2_score(frame_test, frame_pred)
        overall_nrmse = nrmse(frame_pred, frame_test)
        
        print(f"┌─ OVERALL PERFORMANCE ──────────────────────────────────────────────────────────┐")
        print(f"│ RMSE: {overall_rmse:6.3f} ({overall_nrmse:4.1f}%)   │   MAE: {overall_mae:6.3f}   │   R²: {overall_r2:6.3f}              │")
        print(f"└────────────────────────────────────────────────────────────────────────────────┘")
        
        print(f"\n📍 BY SENSOR LOCATION:")
        print("─" * 100)
        
        loc_width = 25
        rmse_width = 7
        pct_width = 6
        mae_width = 7
        r2_width = 7
        actual_width = 7
        pred_width = 6
        
        print(f"{'Location':<{loc_width}}│{'RMSE':<{rmse_width}}│{'%':<{pct_width}}│{'MAE':<{mae_width}}│{'R²':<{r2_width}}│{'Actual':<{actual_width}}│{'Pred':<{pred_width}}")
        print("─" * loc_width + "┼" + "─" * rmse_width + "┼" + "─" * pct_width + "┼" + "─" * mae_width + "┼" + "─" * r2_width + "┼" + "─" * actual_width + "┼" + "─" * pred_width)
        
        sensor_stats = []
        for sensor_idx, sensor in enumerate(sensor_names):
            sensor_pred = y_pred[:, frame, sensor_idx]
            sensor_test = y_test[:, frame, sensor_idx]
            
            sensor_rmse = rmse(sensor_pred, sensor_test)
            sensor_mae = mae(sensor_pred, sensor_test)
            sensor_r2 = r2_score(sensor_test, sensor_pred)
            sensor_nrmse = nrmse(sensor_pred, sensor_test)
            sensor_mean_actual = np.mean(sensor_test)
            sensor_mean_pred = np.mean(sensor_pred)
            
            sensor_stats.append({
                'name': sensor,
                'rmse': sensor_rmse,
                'nrmse': sensor_nrmse,
                'mae': sensor_mae,
                'r2': sensor_r2,
                'mean_actual': sensor_mean_actual,
                'mean_pred': sensor_mean_pred
            })
            
            display_name = sensor[:loc_width-1] if len(sensor) > loc_width-1 else sensor
            
            print(f"{display_name:<{loc_width}}│{sensor_rmse:<{rmse_width}.2f}│{sensor_nrmse:<{pct_width}.1f}%│{sensor_mae:<{mae_width}.2f}│{sensor_r2:<{r2_width}.3f}│{sensor_mean_actual:<{actual_width}.1f}│{sensor_mean_pred:<{pred_width}.1f}")
        
        best_sensor = min(sensor_stats, key=lambda x: x['rmse'])
        worst_sensor = max(sensor_stats, key=lambda x: x['rmse'])
        
        print("─" * 100)
        print(f"🏆 BEST:  {best_sensor['name']:<30} (RMSE: {best_sensor['rmse']:6.3f})")
        print(f"💥 WORST: {worst_sensor['name']:<30} (RMSE: {worst_sensor['rmse']:6.3f})")
        
        excellent = [s for s in sensor_stats if s['rmse'] < 1.5]
        good = [s for s in sensor_stats if 1.5 <= s['rmse'] < 3.0]
        fair = [s for s in sensor_stats if 3.0 <= s['rmse'] < 5.0]
        poor = [s for s in sensor_stats if s['rmse'] >= 5.0]
        
        print(f"\n📊 PERFORMANCE BREAKDOWN:")
        print(f"   🟢 Excellent (RMSE < 1.5): {len(excellent)} sensors")
        print(f"   🟡 Good (1.5-3.0):         {len(good)} sensors") 
        print(f"   🟠 Fair (3.0-5.0):         {len(fair)} sensors")
        print(f"   🔴 Poor (≥ 5.0):           {len(poor)} sensors")
        
        if frame < num_frames - 1:
            print("\n" + "▼" * 100)

def comprehensive_frame_analysis(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    sensor_names: List[str]
) -> None:
    """
    Complete ConvLSTM performance analysis across spatial and temporal dimensions.
    
    This analysis is specifically designed for ConvLSTM models that predict multiple future time steps.
    It evaluates how well the model captures spatio-temporal patterns in air quality data by examining:
    
    1. Temporal degradation: How prediction accuracy decreases with longer prediction frames
    2. Spatial patterns: Which monitoring locations are consistently challenging to predict
    3. Spatio-temporal interactions: How errors vary across both space and time dimensions
    4. Detailed diagnostics: Comprehensive metrics for model improvement guidance
    
    Key ConvLSTM insights revealed:
    - Error accumulation over time frames
    - Spatial aware in prediction difficulty to figure out location speiffic challenges
    - Station-hour combinations
    - Overall model reliability across the spatio-temporal prediction domain
    """
    if len(y_pred.shape) != 3:
        print("Comprehensive frame analysis requires multi-frame predictions")
        return
    
    print("🎯 COMPREHENSIVE CONVLSTM PERFORMANCE ANALYSIS")
    print("=" * 100)
    print(f"📊 Analyzing {y_pred.shape[1]} frames (hours) across {len(sensor_names)} sensors")
    print(f"📈 Dataset: {y_pred.shape[0]} samples")
    print("\nThis analysis evaluates ConvLSTM's ability to predict air quality across multiple future time frames")
    print("and spatial locations, revealing critical spatio-temporal prediction patterns.\n")
    
    # 1. Frame-by-frame RMSE foundation
    print("📊 1. FRAME-BY-FRAME RMSE")
    print("─" * 60)
    plot_frame_by_frame_rmse(y_pred, y_test)
    
    # 2. Average RMSE per station
    print("\n📍 2. AVERAGE RMSE PER STATION (ALL HOURS)")
    print("─" * 55)
    plot_avg_rmse_per_station(y_pred, y_test, sensor_names)
    
    # 3. Summary table
    print_summary_table(y_pred, y_test, sensor_names)
    
    # 4. Heatmap overview
    print("\n🔥 3. RMSE HEATMAP (OVERVIEW)")
    print("─" * 40)
    plot_frame_heatmap(y_pred, y_test, sensor_names)
    
    # 5. Time series for each frame
    print("\n📈 4. TIME SERIES BY FRAME")
    print("─" * 35)
    print("Temporal pattern analysis")
    for frame_idx in range(y_pred.shape[1]):
        print(f"\n🕐 Hour {frame_idx + 1} Time Series")
        plot_frame_time_series(y_pred, y_test, sensor_names, frame_idx)
    
    # 6. Scatter plots for each frame
    print("\n🎯 5. SCATTER PLOTS BY FRAME")
    print("─" * 35)
    print("Prediction accuracy assessment for each time frame:")
    for frame_idx in range(y_pred.shape[1]):
        print(f"\n🎯 Hour {frame_idx + 1} Scatter Plot")
        plot_frame_scatter(y_pred, y_test, frame_idx)
    
    # 7. Detailed statistics
    print_detailed_frame_stats(y_pred, y_test, sensor_names)
    print("✅ COMPREHENSIVE CONVLSTM ANALYSIS COMPLETE!")