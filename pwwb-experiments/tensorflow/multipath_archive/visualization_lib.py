import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class ModelVisualizer:
    """Comprehensive visualization class for ConvLSTM model results."""
    
    def __init__(self, experiment_type='grid', start_date="2023-08-02-00", 
                 end_date="2025-08-02-00", forecast_horizon=5, max_display_frames=5,
                 sensor_names=None):
        """
        Initialize visualizer with experiment configuration.
        
        Args:
            experiment_type: 'grid' or 'sensor'
            start_date: Start date string
            end_date: End date string
            forecast_horizon: Number of forecast frames
            max_display_frames: Maximum frames to display in grid plots (default 5)
            sensor_names: List of sensor names for display purposes
        """
        self.experiment_type = experiment_type
        self.start_date = start_date
        self.end_date = end_date
        self.forecast_horizon = forecast_horizon
        self.max_display_frames = max_display_frames
        self.sensor_names = sensor_names or []
        
        self.COLOR_PRED = 'blue'
        self.COLOR_TRUE = 'orange'
        
        self.setup_dates()
    
    def setup_dates(self):
        """Setup date ranges for time series plots."""
        start = datetime.strptime(self.start_date, "%Y-%m-%d-%H")
        end = datetime.strptime(self.end_date, "%Y-%m-%d-%H")
        self.dates = pd.date_range(start=start, end=end, freq='h')
        
        total_timesteps = len(self.dates)
        usable_timesteps = total_timesteps - self.forecast_horizon
        train_size = int(usable_timesteps * 0.7)
        valid_size = int(usable_timesteps * 0.15)
        self.y_test_start_idx = train_size + valid_size + self.forecast_horizon * 2 - 1
    
    def calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error."""
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true != 0)
        if valid_mask.sum() == 0:
            return np.nan
        
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        mape = np.mean(np.abs((y_true_valid - y_pred_valid) / y_true_valid)) * 100
        return mape
    
    def calculate_nrmse(self, y_true, y_pred):
        """Calculate Normalized RMSE."""
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if valid_mask.sum() == 0:
            return np.nan
        
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
        mean_val = np.mean(y_true_valid)
        
        if mean_val > 0:
            nrmse = (rmse / mean_val) * 100
        else:
            nrmse = rmse
        
        return nrmse
    
    def plot_training_history(self, history, model_name="Model"):
        """
        Plot training and validation loss history.
        
        Args:
            history: Keras training history object
            model_name: Name for the plot title
        
        Returns:
            fig: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        best_val_loss = min(val_loss)
        best_epoch = val_loss.index(best_val_loss) + 1
        
        ax.plot(train_loss, label='Training Loss', linewidth=2)
        ax.plot(val_loss, label='Validation Loss', linewidth=2)
        ax.axhline(y=best_val_loss, color='r', linestyle='--', alpha=0.5, 
                   label=f'Best Val Loss: {best_val_loss:.4f}')
        ax.axvline(x=best_epoch-1, color='g', linestyle='--', alpha=0.3,
                   label=f'Best Epoch: {best_epoch}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MAE)')
        ax.set_title(f'Training History - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        final_train = train_loss[-1]
        final_val = val_loss[-1]
        textstr = f'Final Train: {final_train:.4f}\nFinal Valid: {final_val:.4f}\nBest Valid: {best_val_loss:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.7, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig
    
    def plot_averaged_grids(self, y_pred, Y_test, sample_range="all"):
        """
        Plot averaged prediction grids across test samples.
        
        Args:
            y_pred: Predictions (samples, frames, height, width) or (samples, frames, height, width, channels)
            Y_test: Ground truth (samples, frames, height, width) or (samples, frames, height, width, channels)
            sample_range: "all", "best_week", or "worst_week"
        
        Returns:
            fig: Matplotlib figure object
        """
        if y_pred.ndim == 5:
            y_pred = y_pred[..., 0]
        if Y_test.ndim == 5:
            Y_test = Y_test[..., 0]
        
        if sample_range == "best_week":
            start_idx, end_idx = self._find_best_week(y_pred, Y_test)
            title_suffix = f"Best Week ({end_idx-start_idx} samples)"
        elif sample_range == "worst_week":
            start_idx, end_idx = self._find_worst_week(y_pred, Y_test)
            title_suffix = f"Worst Week ({end_idx-start_idx} samples)"
        else:
            start_idx, end_idx = 0, len(y_pred)
            title_suffix = f"All Test Samples ({len(y_pred)} samples)"
        
        avg_pred = np.nanmean(y_pred[start_idx:end_idx], axis=0)
        avg_true = np.nanmean(Y_test[start_idx:end_idx], axis=0)
        
        overall_rmse = np.sqrt(np.nanmean((avg_pred - avg_true)**2))
        overall_mae = np.nanmean(np.abs(avg_pred - avg_true))
        
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f'Averaged Grids - {title_suffix}\nRMSE: {overall_rmse:.3f}, MAE: {overall_mae:.3f}', 
                    fontsize=14, fontweight='bold')
        
        frames_to_show = min(self.max_display_frames, self.forecast_horizon)
        gs = fig.add_gridspec(3, frames_to_show + 1,
                            width_ratios=[1]*frames_to_show + [0.05],
                            height_ratios=[1, 1, 1],
                            hspace=0.15, wspace=0.1)
        
        vmin = min(np.nanmin(avg_pred), np.nanmin(avg_true))
        vmax = max(np.nanmax(avg_pred), np.nanmax(avg_true))
        max_error = np.nanmax(np.abs(avg_pred - avg_true))
        
        for frame in range(frames_to_show):
            ax = fig.add_subplot(gs[0, frame])
            im = ax.imshow(avg_pred[frame], cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f't+{frame+1}', fontsize=10)
            ax.axis('off')
            if frame == 0:
                ax.text(-0.3, 0.5, 'Averaged\nPredictions',
                    transform=ax.transAxes, fontsize=11, fontweight='bold',
                    va='center', ha='right')
            
            ax = fig.add_subplot(gs[1, frame])
            im = ax.imshow(avg_true[frame], cmap='viridis', vmin=vmin, vmax=vmax)
            ax.axis('off')
            if frame == 0:
                ax.text(-0.3, 0.5, 'Averaged\nTruth',
                    transform=ax.transAxes, fontsize=11, fontweight='bold',
                    va='center', ha='right')
            
            ax = fig.add_subplot(gs[2, frame])
            error = avg_pred[frame] - avg_true[frame]
            im_error = ax.imshow(error, cmap='RdBu_r', vmin=-max_error, vmax=max_error)
            mae = np.nanmean(np.abs(error))
            ax.set_title(f'MAE: {mae:.2f}', fontsize=9)
            ax.axis('off')
            if frame == 0:
                ax.text(-0.3, 0.5, 'Averaged\nError',
                    transform=ax.transAxes, fontsize=11, fontweight='bold',
                    va='center', ha='right')
        
        cbar_ax1 = fig.add_axes([0.92, 0.45, 0.02, 0.35])
        plt.colorbar(im, cax=cbar_ax1, label='PM2.5 (μg/m³)')
        
        cbar_ax2 = fig.add_axes([0.92, 0.08, 0.02, 0.25])
        plt.colorbar(im_error, cax=cbar_ax2, label='Error (μg/m³)')
        
        return fig
    
    def plot_time_series(self, y_pred, Y_test, n_sensors=None, plot_type="sensor_avg"):
        """
        Plot time series comparisons.
        
        Args:
            y_pred: Predictions
            Y_test: Ground truth
            n_sensors: Number of sensors (for sensor experiments)
            plot_type: "sensor_avg", "grid_avg", "best_worst_weeks"
        
        Returns:
            fig: Matplotlib figure object
        """
        if plot_type == "sensor_avg":
            return self._plot_sensor_time_series(y_pred, Y_test, n_sensors)
        elif plot_type == "grid_avg":
            return self._plot_grid_time_series(y_pred, Y_test)
        elif plot_type == "best_worst_weeks":
            return self._plot_best_worst_weeks(y_pred, Y_test, n_sensors)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
    
    def _plot_sensor_time_series(self, y_pred, Y_test, n_sensors):
        """Plot sensor-averaged time series with MAPE scores."""
        if len(y_pred.shape) == 3:
            y_pred_avg = np.nanmean(y_pred, axis=2)
            Y_test_avg = np.nanmean(Y_test, axis=2)
        else:
            y_pred_avg = np.nanmean(y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1), axis=2)
            Y_test_avg = np.nanmean(Y_test.reshape(Y_test.shape[0], Y_test.shape[1], -1), axis=2)
        
        fig, axes = plt.subplots(2, 1, figsize=(20, 12))
        
        series_pred = y_pred_avg[:, 0]
        series_test = Y_test_avg[:, 0]
        
        axes[0].plot(range(len(series_pred)), series_pred, label='Predicted', 
                    color=self.COLOR_PRED, alpha=0.7, linewidth=1)
        axes[0].plot(range(len(series_test)), series_test, label='Actual', 
                    color=self.COLOR_TRUE, alpha=0.7, linewidth=1)
        
        title = f'First Frame (t+1) - Average Across Sensors'
        if n_sensors:
            title = f'First Frame (t+1) - Average Across {n_sensors} Sensors'
        axes[0].set_title(title, fontsize=14)
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('PM2.5 (μg/m³)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        rmse = np.sqrt(mean_squared_error(series_test, series_pred))
        mae = mean_absolute_error(series_test, series_pred)
        mape = self.calculate_mape(series_test, series_pred)
        r2 = r2_score(series_test, series_pred)
        textstr = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%\nR²: {r2:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axes[0].text(0.02, 0.98, textstr, transform=axes[0].transAxes, 
                    fontsize=10, verticalalignment='top', bbox=props)
        
        y_pred_all = np.nanmean(y_pred_avg, axis=1)
        Y_test_all = np.nanmean(Y_test_avg, axis=1)
        
        axes[1].plot(range(len(y_pred_all)), y_pred_all, label='Predicted',
                    color=self.COLOR_PRED, alpha=0.7, linewidth=1)
        axes[1].plot(range(len(Y_test_all)), Y_test_all, label='Actual',
                    color=self.COLOR_TRUE, alpha=0.7, linewidth=1)
        
        axes[1].set_title(f'All {self.forecast_horizon} Frames Averaged', fontsize=14)
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('PM2.5 (μg/m³)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        rmse_all = np.sqrt(mean_squared_error(Y_test_all, y_pred_all))
        mae_all = mean_absolute_error(Y_test_all, y_pred_all)
        mape_all = self.calculate_mape(Y_test_all, y_pred_all)
        r2_all = r2_score(Y_test_all, y_pred_all)
        textstr = f'RMSE: {rmse_all:.2f}\nMAE: {mae_all:.2f}\nMAPE: {mape_all:.2f}%\nR²: {r2_all:.3f}'
        axes[1].text(0.02, 0.98, textstr, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig
    
    def _plot_grid_time_series(self, y_pred, Y_test):
        """Plot grid-averaged time series."""
        y_pred_avg = np.nanmean(y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1), axis=2)
        Y_test_avg = np.nanmean(Y_test.reshape(Y_test.shape[0], Y_test.shape[1], -1), axis=2)
        
        fig, ax = plt.subplots(figsize=(20, 6))
        
        series_pred = y_pred_avg[:, 0]
        series_test = Y_test_avg[:, 0]
        
        ax.plot(range(len(series_pred)), series_pred, label='Predicted',
                color=self.COLOR_PRED, alpha=0.7, linewidth=1)
        ax.plot(range(len(series_test)), series_test, label='Actual',
                color=self.COLOR_TRUE, alpha=0.7, linewidth=1)
        
        ax.set_title('Grid-Level: First Frame (t+1) - Average Across All Grid Points', fontsize=14)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('PM2.5 (μg/m³)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        rmse = np.sqrt(mean_squared_error(series_test, series_pred))
        mae = mean_absolute_error(series_test, series_pred)
        mape = self.calculate_mape(series_test, series_pred)
        r2 = r2_score(series_test, series_pred)
        textstr = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%\nR²: {r2:.3f}'
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig
    
    def _plot_best_worst_weeks(self, y_pred, Y_test, n_sensors, sensor_locations=None):
        """Plot best and worst performing weeks/months for all 5 forecast frames."""
        import matplotlib.pyplot as plt
        import os
        
        if hasattr(self, 'save_path') and self.save_path:
            output_dir = f"{self.save_path}/best_worst_analysis"
        else:
            output_dir = "best_worst_analysis"
        
        os.makedirs(output_dir, exist_ok=True)
        
        if len(y_pred.shape) > 3:
            y_pred_avg = np.nanmean(y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1), axis=2)
            Y_test_avg = np.nanmean(Y_test.reshape(Y_test.shape[0], Y_test.shape[1], -1), axis=2)
        else:
            y_pred_avg = y_pred
            Y_test_avg = Y_test
        
        def find_best_worst_overall(data_pred, data_true, period_hours):
            """Find best and worst periods based on average performance across ALL frames."""
            best_nrmse = np.inf
            worst_nrmse = 0
            best_idx = 0
            worst_idx = 0
            
            for start_idx in range(0, max(1, len(data_true) - period_hours), 24):
                end_idx = min(start_idx + period_hours, len(data_true))
                
                period_nrmses = []
                for frame in range(5):
                    pred_frame = data_pred[start_idx:end_idx, frame]
                    true_frame = data_true[start_idx:end_idx, frame]
                    
                    valid_mask = ~np.isnan(true_frame) & ~np.isnan(pred_frame)
                    if valid_mask.sum() > 0:
                        rmse = np.sqrt(np.mean((true_frame[valid_mask] - pred_frame[valid_mask])**2))
                        mean_val = np.mean(true_frame[valid_mask])
                        if mean_val > 0:
                            nrmse = (rmse / mean_val) * 100
                            period_nrmses.append(nrmse)
                
                if period_nrmses:
                    avg_nrmse = np.mean(period_nrmses)
                    if avg_nrmse < best_nrmse:
                        best_nrmse = avg_nrmse
                        best_idx = start_idx
                    if avg_nrmse > worst_nrmse and avg_nrmse < np.inf:
                        worst_nrmse = avg_nrmse
                        worst_idx = start_idx
            
            return best_idx, worst_idx
        
        print("    Finding best/worst weeks based on overall performance...")
        week_hours = 7 * 24
        
        best_week_idx, worst_week_idx = find_best_worst_overall(y_pred_avg, Y_test_avg, week_hours)
        
        print(f"      Best week starts at index {best_week_idx}")
        print(f"      Worst week starts at index {worst_week_idx}")
        
        for frame in range(5):
            fig, axes = plt.subplots(2, 1, figsize=(18, 10))
            
            best_end = min(best_week_idx + week_hours, len(y_pred_avg))
            try:
                dates_best = self.dates[self.y_test_start_idx + best_week_idx : 
                                    self.y_test_start_idx + best_end]
            except:
                dates_best = range(best_end - best_week_idx)
            
            pred_best = y_pred_avg[best_week_idx:best_end, frame]
            true_best = Y_test_avg[best_week_idx:best_end, frame]
            
            best_rmse = np.sqrt(np.mean((true_best - pred_best)**2))
            best_mae = np.mean(np.abs(true_best - pred_best))
            best_mape = self.calculate_mape(true_best, pred_best)
            best_nrmse = self.calculate_nrmse(true_best, pred_best)
            
            axes[0].plot(dates_best, pred_best, label='Predicted', color='blue', linewidth=2)
            axes[0].plot(dates_best, true_best, label='Actual', color='orange', linewidth=2)
            axes[0].set_ylabel("PM2.5 (μg/m³)")
            axes[0].set_title(f"Best Week - Grid Average - NRMSE: {best_nrmse:.2f}%, MAPE: {best_mape:.2f}%, RMSE: {best_rmse:.2f}")
            axes[0].legend(loc="upper right")
            axes[0].grid(True, alpha=0.3)
            
            worst_end = min(worst_week_idx + week_hours, len(y_pred_avg))
            try:
                dates_worst = self.dates[self.y_test_start_idx + worst_week_idx : 
                                        self.y_test_start_idx + worst_end]
            except:
                dates_worst = range(worst_end - worst_week_idx)
            
            pred_worst = y_pred_avg[worst_week_idx:worst_end, frame]
            true_worst = Y_test_avg[worst_week_idx:worst_end, frame]
            
            worst_rmse = np.sqrt(np.mean((true_worst - pred_worst)**2))
            worst_mae = np.mean(np.abs(true_worst - pred_worst))
            worst_mape = self.calculate_mape(true_worst, pred_worst)
            worst_nrmse = self.calculate_nrmse(true_worst, pred_worst)
            
            axes[1].plot(dates_worst, pred_worst, label='Predicted', color='blue', linewidth=2)
            axes[1].plot(dates_worst, true_worst, label='Actual', color='orange', linewidth=2)
            axes[1].set_xlabel("Time (hourly)")
            axes[1].set_ylabel("PM2.5 (μg/m³)")
            axes[1].set_title(f"Worst Week - Grid Average - NRMSE: {worst_nrmse:.2f}%, MAPE: {worst_mape:.2f}%, RMSE: {worst_rmse:.2f}")
            axes[1].legend(loc="upper right")
            axes[1].grid(True, alpha=0.3)
            
            fig.suptitle(f'Best and Worst Week Predictions (Hour {frame+1} Forecast) - Grid Average Performance', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/best_worst_week_hour{frame+1}.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"      Saved Hour {frame+1} weekly analysis")
        
        print("    Finding best/worst months based on overall performance...")
        month_hours = 30 * 24
        
        if len(Y_test_avg) >= month_hours:
            best_month_idx, worst_month_idx = find_best_worst_overall(y_pred_avg, Y_test_avg, month_hours)
            
            print(f"      Best month starts at index {best_month_idx}")
            print(f"      Worst month starts at index {worst_month_idx}")
            
            for frame in range(5):
                fig, axes = plt.subplots(2, 1, figsize=(20, 10))
                
                best_end = min(best_month_idx + month_hours, len(y_pred_avg))
                try:
                    dates_best = self.dates[self.y_test_start_idx + best_month_idx : 
                                        self.y_test_start_idx + best_end]
                except:
                    dates_best = range(best_end - best_month_idx)
                
                pred_best = y_pred_avg[best_month_idx:best_end, frame]
                true_best = Y_test_avg[best_month_idx:best_end, frame]
                
                best_rmse = np.sqrt(np.mean((true_best - pred_best)**2))
                best_mae = np.mean(np.abs(true_best - pred_best))
                best_mape = self.calculate_mape(true_best, pred_best)
                best_nrmse = self.calculate_nrmse(true_best, pred_best)
                
                axes[0].plot(dates_best, pred_best, label='Predicted', color='blue', linewidth=2)
                axes[0].plot(dates_best, true_best, label='Actual', color='orange', linewidth=2)
                axes[0].set_ylabel("PM2.5 (μg/m³)")
                axes[0].set_title(f"Best Month - Grid Average - NRMSE: {best_nrmse:.2f}%, MAPE: {best_mape:.2f}%, RMSE: {best_rmse:.2f}")
                axes[0].legend(loc="upper right")
                axes[0].grid(True, alpha=0.3)
                
                worst_end = min(worst_month_idx + month_hours, len(y_pred_avg))
                try:
                    dates_worst = self.dates[self.y_test_start_idx + worst_month_idx : 
                                            self.y_test_start_idx + worst_end]
                except:
                    dates_worst = range(worst_end - worst_month_idx)
                
                pred_worst = y_pred_avg[worst_month_idx:worst_end, frame]
                true_worst = Y_test_avg[worst_month_idx:worst_end, frame]
                
                worst_rmse = np.sqrt(np.mean((true_worst - pred_worst)**2))
                worst_mae = np.mean(np.abs(true_worst - pred_worst))
                worst_mape = self.calculate_mape(true_worst, pred_worst)
                worst_nrmse = self.calculate_nrmse(true_worst, pred_worst)
                
                axes[1].plot(dates_worst, pred_worst, label='Predicted', color='blue', linewidth=2)
                axes[1].plot(dates_worst, true_worst, label='Actual', color='orange', linewidth=2)
                axes[1].set_xlabel("Time (hourly)")
                axes[1].set_ylabel("PM2.5 (μg/m³)")
                axes[1].set_title(f"Worst Month - Grid Average - NRMSE: {worst_nrmse:.2f}%, MAPE: {worst_mape:.2f}%, RMSE: {worst_rmse:.2f}")
                axes[1].legend(loc="upper right")
                axes[1].grid(True, alpha=0.3)
                
                fig.suptitle(f'Best and Worst Month Predictions (Hour {frame+1} Forecast) - Grid Average Performance', 
                            fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/best_worst_month_hour{frame+1}.png", dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"      Saved Hour {frame+1} monthly analysis")
        else:
            print("      Not enough data for monthly analysis")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, f'Best/Worst Analysis Complete\nSaved to: {output_dir}',
            ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    def plot_sample_inputs(self, X_test, input_channels, n_samples=3):
        """
        Visualize random input samples.
        
        Args:
            X_test: Test input data (samples, frames, height, width, channels)
            input_channels: List of channel names
            n_samples: Number of random samples to show
        
        Returns:
            fig: Matplotlib figure object
        """
        import random
        
        random_indices = random.sample(range(min(100, len(X_test))), n_samples)
        
        figs = []
        for idx_num, sample_idx in enumerate(random_indices):
            n_frames = X_test.shape[1]
            n_channels = len(input_channels)
            
            fig, axes = plt.subplots(n_channels, n_frames,
                                    figsize=(n_frames*2.5, n_channels*2.5))
            
            if n_channels == 1 and n_frames == 1:
                axes = np.array([[axes]])
            elif n_channels == 1:
                axes = axes.reshape(1, -1)
            elif n_frames == 1:
                axes = axes.reshape(-1, 1)
            
            fig.suptitle(f'Sample {sample_idx} - Input Channels ({n_frames} timesteps)',
                        fontsize=16, y=1.02)
            
            for ch_idx, channel_name in enumerate(input_channels):
                channel_data = X_test[sample_idx, :, :, :, ch_idx]
                vmin = np.min(channel_data)
                vmax = np.max(channel_data)
                
                for t_idx in range(n_frames):
                    ax = axes[ch_idx, t_idx] if n_channels > 1 else axes[0, t_idx]
                    
                    data = X_test[sample_idx, t_idx, :, :, ch_idx]
                    im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
                    
                    if t_idx == 0:
                        ax.set_ylabel(channel_name, fontsize=10, fontweight='bold')
                    if ch_idx == 0:
                        ax.set_title(f't-{n_frames-t_idx}', fontsize=10)
                    
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    if t_idx == n_frames - 1:
                        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        cbar.ax.tick_params(labelsize=8)
            
            plt.tight_layout()
            figs.append(fig)
        
        return figs
    
    def create_summary_report(self, y_pred, Y_test, history=None, model_name="Model",
                            n_sensors=None, save_path=None, sensor_names=None):
        """
        Create a comprehensive summary report with all visualizations.
        
        Args:
            y_pred: Model predictions
            Y_test: Ground truth
            history: Keras training history (optional)
            model_name: Name of the model
            n_sensors: Number of sensors
            save_path: Path to save figures (optional)
            sensor_names: List of sensor names for labeling
        
        Returns:
            Dictionary of figure objects
        """
        self.save_path = save_path
        if sensor_names:
            self.sensor_names = sensor_names
        
        figures = {}
        
        print(f"\n{'='*60}")
        print(f"GENERATING VISUALIZATION REPORT FOR {model_name}")
        print(f"{'='*60}")
        
        if history is not None:
            print("Plotting training history...")
            figures['training_history'] = self.plot_training_history(history, model_name)
            if save_path:
                figures['training_history'].savefig(f"{save_path}/training_history.png", dpi=100, bbox_inches='tight')
        
        if self.experiment_type == "grid":
            print("Plotting averaged grids...")
            figures['avg_grids_all'] = self.plot_averaged_grids(y_pred, Y_test, "all")
            if save_path:
                figures['avg_grids_all'].savefig(f"{save_path}/averaged_grids_all.png", dpi=100, bbox_inches='tight')
            
            figures['avg_grids_best'] = self.plot_averaged_grids(y_pred, Y_test, "best_week")
            if save_path:
                figures['avg_grids_best'].savefig(f"{save_path}/averaged_grids_best.png", dpi=100, bbox_inches='tight')
            
            figures['avg_grids_worst'] = self.plot_averaged_grids(y_pred, Y_test, "worst_week")
            if save_path:
                figures['avg_grids_worst'].savefig(f"{save_path}/averaged_grids_worst.png", dpi=100, bbox_inches='tight')
        
        print("Plotting time series...")
        figures['time_series'] = self.plot_time_series(y_pred, Y_test, n_sensors, "sensor_avg")
        if save_path:
            figures['time_series'].savefig(f"{save_path}/time_series.png", dpi=100, bbox_inches='tight')
        
        print("Plotting best/worst weeks with sensor names and MAPE...")
        figures['best_worst_weeks'] = self.plot_time_series(y_pred, Y_test, n_sensors, "best_worst_weeks")
        if save_path:
            figures['best_worst_weeks'].savefig(f"{save_path}/best_worst_weeks.png", dpi=100, bbox_inches='tight')
        
        print(f"Report generation complete! Generated {len(figures)} figures.")
        return figures
    
    def _find_best_week(self, y_pred, Y_test):
        """Find the best performing week in predictions."""
        best_nrmse = np.inf
        best_idx = 0
        
        for start_idx in range(0, max(1, len(Y_test) - 168), 24):
            week_end = min(start_idx + 168, len(Y_test))
            week_pred = y_pred[start_idx:week_end].flatten()
            week_true = Y_test[start_idx:week_end].flatten()
            
            mask = ~np.isnan(week_true)
            if mask.sum() > 0:
                rmse = np.sqrt(mean_squared_error(week_true[mask], week_pred[mask]))
                range_val = week_true[mask].max() - week_true[mask].min()
                if range_val > 0:
                    nrmse = (rmse / range_val) * 100
                    if nrmse < best_nrmse:
                        best_nrmse = nrmse
                        best_idx = start_idx
        
        return best_idx, min(best_idx + 168, len(Y_test))
    
    def _find_worst_week(self, y_pred, Y_test):
        """Find the worst performing week in predictions."""
        worst_nrmse = 0
        worst_idx = 0
        
        for start_idx in range(0, max(1, len(Y_test) - 168), 24):
            week_end = min(start_idx + 168, len(Y_test))
            week_pred = y_pred[start_idx:week_end].flatten()
            week_true = Y_test[start_idx:week_end].flatten()
            
            mask = ~np.isnan(week_true)
            if mask.sum() > 0:
                rmse = np.sqrt(mean_squared_error(week_true[mask], week_pred[mask]))
                range_val = week_true[mask].max() - week_true[mask].min()
                if range_val > 0:
                    nrmse = (rmse / range_val) * 100
                    if nrmse > worst_nrmse and nrmse != np.inf:
                        worst_nrmse = nrmse
                        worst_idx = start_idx
        
        return worst_idx, min(worst_idx + 168, len(Y_test))
    
    def _calculate_week_nrmse(self, y_true, y_pred, start_idx):
        """Calculate NRMSE for a week of data."""
        end_idx = min(start_idx + 168, len(y_true))
        
        week_true = y_true[start_idx:end_idx]
        week_pred = y_pred[start_idx:end_idx]
        
        if len(week_true.shape) > 1:
            week_true = week_true.flatten()
            week_pred = week_pred.flatten()
        
        mask = ~np.isnan(week_true)
        if mask.sum() == 0:
            return np.inf
        
        rmse = np.sqrt(mean_squared_error(week_true[mask], week_pred[mask]))
        range_val = np.nanmax(week_true) - np.nanmin(week_true)
        
        if range_val == 0:
            return np.inf
        
        return (rmse / range_val) * 100
    
    def _calculate_week_nrmse_sensor(self, y_true, y_pred, start_idx):
        """Calculate NRMSE for a week of data for a specific sensor."""
        end_idx = min(start_idx + 168, len(y_true))
        
        week_true = y_true[start_idx:end_idx]
        week_pred = y_pred[start_idx:end_idx]
        
        mask = ~np.isnan(week_true) & ~np.isnan(week_pred)
        if mask.sum() == 0:
            return np.inf
        
        week_true_valid = week_true[mask]
        week_pred_valid = week_pred[mask]
        
        rmse = np.sqrt(mean_squared_error(week_true_valid, week_pred_valid))
        mean_val = np.mean(week_true_valid)
        
        if mean_val > 0:
            return (rmse / mean_val) * 100
        else:
            return np.inf