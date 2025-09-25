import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import calendar
from scipy.interpolate import griddata, interp1d
from scipy.ndimage import zoom
import earthaccess
from dotenv import load_dotenv
import time
import warnings
warnings.filterwarnings('ignore')

class TempoL3Processor:
    def __init__(
        self,
        start_date="2023-08-02",
        end_date="2025-08-02",
        extent=(-118.75, -117.5, 33.5, 34.5),
        dim=40,
        cache_dir='data/tempo_l3_cache/',
        output_dir='data/tempo_l3_processed/',
        filling_strategy='persistence',
        decay_hours=6,
        max_retries=10,
        initial_wait=5,
        verbose=True,
        test_mode=False
    ):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.test_mode = test_mode
        
        if self.test_mode:
            self.end_date = self.start_date + timedelta(days=7)
            print(f"TEST MODE: Processing only 1 week of data")
        
        self.extent = extent
        self.lon_min, self.lon_max, self.lat_min, self.lat_max = extent
        self.dim = dim
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.filling_strategy = filling_strategy
        self.decay_hours = decay_hours
        self.max_retries = max_retries
        self.initial_wait = initial_wait
        self.verbose = verbose
        
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        self._setup_auth()
        
        self.continuous_hourly_data = []
        self.continuous_timestamps = []
        
        self.all_timestamps = self._generate_all_timestamps()
        
        print(f"TEMPO L3 Hourly Data Processor")
        print(f"=" * 70)
        print(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Total hours to process: {len(self.all_timestamps)}")
        print(f"First timestamp: {self.all_timestamps[0] if self.all_timestamps else 'None'}")
        print(f"Last timestamp: {self.all_timestamps[-1] if self.all_timestamps else 'None'}")
        print(f"Extent: {self.extent} (LA County)")
        print(f"Output resolution: {self.dim}×{self.dim}")
        print(f"Filling strategy: {self.filling_strategy}")
        if self.test_mode:
            print(f"TEST MODE: Processing only 1 week of data")
        print(f"=" * 70)
    
    def _generate_all_timestamps(self):
        timestamps = []
        current = self.start_date
        
        while current < self.end_date:
            timestamps.append(current)
            current += timedelta(hours=1)
        
        return timestamps
    
    def _setup_auth(self):
        env_paths = ['.env', '../.env', '../libs/.env']
        for env_path in env_paths:
            if os.path.exists(env_path):
                load_dotenv(env_path)
                break
        
        try:
            earthaccess.login()
            if self.verbose:
                print("NASA Earthdata authenticated")
        except Exception as e:
            print(f"Authentication failed: {e}")
            raise
    
    def _download_with_retry(self, download_function, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                result = download_function(*args, **kwargs)
                if result is not None:
                    return result
            except Exception as e:
                wait_time = self.initial_wait * (2 ** attempt)
                print(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"All {self.max_retries} attempts failed")
                    raise
        
        return None
    
    def process_all_months(self):
        last_valid_reading = None
        last_valid_time = None
        
        for timestamp in self.all_timestamps:
            self.continuous_timestamps.append(timestamp)
        
        from itertools import groupby
        
        monthly_groups = []
        for key, group in groupby(self.all_timestamps, key=lambda x: (x.year, x.month)):
            monthly_groups.append((key, list(group)))
        
        for (year, month), month_timestamps in monthly_groups:
            print(f"\nProcessing {year}-{month:02d}...")
            
            monthly_data = self._process_single_month(year, month, month_timestamps, 
                                                     last_valid_reading, last_valid_time)
            
            if monthly_data:
                if 'last_valid_reading' in monthly_data and monthly_data['last_valid_reading'] is not None:
                    last_valid_reading = monthly_data['last_valid_reading']
                if 'last_valid_time' in monthly_data and monthly_data['last_valid_time'] is not None:
                    last_valid_time = monthly_data['last_valid_time']
                
                for ts, data in zip(month_timestamps, monthly_data['hourly_data']):
                    self.continuous_hourly_data.append(data)
            else:
                print(f"No data available for {year}-{month:02d}, filling with persistence strategy")
                for ts in month_timestamps:
                    filled_array = self._apply_filling_strategy(ts, last_valid_reading, last_valid_time)
                    self.continuous_hourly_data.append(filled_array)
        
        assert len(self.continuous_hourly_data) == len(self.all_timestamps), \
            f"Data mismatch: {len(self.continuous_hourly_data)} data points vs {len(self.all_timestamps)} timestamps"
        
        self._save_hourly_data()
    
    def _process_single_month(self, year, month, month_timestamps, last_valid_reading, last_valid_time):
        month_cache_file = os.path.join(self.cache_dir, f"processed_{year}{month:02d}.npz")
        
        if os.path.exists(month_cache_file):
            print(f"Loading processed month from cache: {month_cache_file}")
            try:
                data = np.load(month_cache_file, allow_pickle=True)
                
                cached_timestamps = list(data['timestamps'])
                if len(cached_timestamps) == len(month_timestamps):
                    match = all(pd.to_datetime(str(ct)) == mt for ct, mt in zip(cached_timestamps, month_timestamps))
                    if match:
                        return {
                            'hourly_data': list(data['hourly_data']),
                            'timestamps': month_timestamps,
                            'last_valid_reading': data['last_valid_reading'],
                            'last_valid_time': pd.to_datetime(str(data['last_valid_time']))
                        }
                print(f"Cache timestamps don't match, reprocessing...")
            except Exception as e:
                print(f"Cache file corrupted, reprocessing: {e}")
        
        monthly_batch = self._download_with_retry(self._download_monthly_batch, year, month, month_timestamps)
        
        if not monthly_batch:
            print(f"Failed to download data for {year}-{month:02d}")
            return None
        
        monthly_hourly_data = []
        local_last_valid = last_valid_reading
        local_last_time = last_valid_time
        
        tempo_data_by_hour = {}
        
        for filepath in monthly_batch['files']:
            try:
                processed = self._process_l3_file(filepath)
                if processed:
                    ts = processed['timestamp']
                    hour_key = ts.replace(minute=0, second=0, microsecond=0)
                    tempo_data_by_hour[hour_key] = processed['array']
            except:
                continue
        
        for ts in month_timestamps:
            hour_key = ts.replace(minute=0, second=0, microsecond=0)
            
            if hour_key in tempo_data_by_hour:
                array_data = tempo_data_by_hour[hour_key]
                monthly_hourly_data.append(array_data)
                local_last_valid = array_data.copy()
                local_last_time = ts
            else:
                filled = self._apply_filling_strategy(ts, local_last_valid, local_last_time)
                monthly_hourly_data.append(filled)
        
        print(f"   🧹 Cleaning up {len(monthly_batch['files'])} NC files...")
        for filepath in monthly_batch['files']:
            try:
                os.remove(filepath) if os.path.exists(filepath) else None
            except:
                pass
        
        try:
            os.rmdir(monthly_batch['month_dir'])
        except:
            pass
        
        if monthly_hourly_data:
            print(f"Saving aligned month to cache...")
            np.savez_compressed(
                month_cache_file,
                hourly_data=monthly_hourly_data,
                timestamps=[str(ts) for ts in month_timestamps],
                last_valid_reading=local_last_valid,
                last_valid_time=str(local_last_time) if local_last_time else None
            )
            
            return {
                'hourly_data': monthly_hourly_data,
                'timestamps': month_timestamps,
                'last_valid_reading': local_last_valid,
                'last_valid_time': local_last_time
            }
        
        return None
    
    def _download_monthly_batch(self, year, month, month_timestamps):
        if not month_timestamps:
            return None
            
        month_start = month_timestamps[0]
        month_end = month_timestamps[-1]
        
        search_start = month_start.replace(hour=0, minute=0, second=0, microsecond=0)
        search_end = month_end.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        print(f"\n🔍 Searching for TEMPO L3 files from {search_start.date()} to {search_end.date()}")
        
        try:
            results = earthaccess.search_data(
                short_name='TEMPO_NO2_L3',
                temporal=(search_start, search_end + timedelta(seconds=1)),
                count=1000
            )
            
            if not results:
                print(f"No TEMPO L3 files found for {year}-{month:02d}")
                return None
            
            print(f"Found {len(results)} L3 files")
            
            month_dir = os.path.join(self.cache_dir, f"temp_{year}{month:02d}")
            os.makedirs(month_dir, exist_ok=True)
            
            print(f"Downloading {len(results)} files...")
            downloaded_files = earthaccess.download(results, local_path=month_dir)
            
            print(f"Downloaded {len(downloaded_files)} files successfully")
            return {
                'month_dir': month_dir,
                'files': downloaded_files,
                'year': year,
                'month': month
            }
            
        except Exception as e:
            print(f"Error downloading monthly batch: {e}")
            return None
    
    def _apply_filling_strategy(self, current_time, last_valid_reading, last_valid_time):
        
        if last_valid_reading is None:
            return np.full((self.dim, self.dim), 0.001, dtype=np.float32)
        
        if self.filling_strategy == 'zero':
            return np.full((self.dim, self.dim), 0.001, dtype=np.float32)
        
        elif self.filling_strategy == 'persistence':
            return last_valid_reading.copy()
        
        elif self.filling_strategy in ['persistence_decay', 'bidirectional_decay']:
            if last_valid_time is None:
                return np.full((self.dim, self.dim), 0.001, dtype=np.float32)
            
            if hasattr(current_time, 'tz') and current_time.tz is not None:
                current_time = current_time.tz_localize(None)
            if hasattr(last_valid_time, 'tz') and last_valid_time.tz is not None:
                last_valid_time = last_valid_time.tz_localize(None)
            
            hours_elapsed = (current_time - last_valid_time).total_seconds() / 3600.0
            
            if hours_elapsed < 0:
                return last_valid_reading.copy()
            
            if hours_elapsed > 24:
                return np.full((self.dim, self.dim), 0.001, dtype=np.float32)
            
            decay_factor = np.exp(-hours_elapsed / self.decay_hours)
            filled_array = last_valid_reading * decay_factor
            
            noise_level = max(0, 0.01 * (1 - decay_factor))
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, filled_array.shape)
                filled_array = filled_array + noise
            
            filled_array = np.maximum(0.001, filled_array)
            
            return filled_array.astype(np.float32)
        
        else:
            raise ValueError(f"Unknown filling strategy: {self.filling_strategy}")
    
    def _process_l3_file(self, filepath):
        try:
            ds_root = xr.open_dataset(filepath)
            lat = ds_root['latitude'].values
            lon = ds_root['longitude'].values
            
            time_str = ds_root.attrs.get('time_coverage_start', '')
            timestamp = pd.to_datetime(time_str)
            if timestamp.tz is not None:
                timestamp = timestamp.tz_localize(None)
            
            ds_root.close()
            
            ds_product = xr.open_dataset(filepath, group='product')
            
            no2_trop = ds_product['vertical_column_troposphere'].values
            qa_flag = ds_product['main_data_quality_flag'].values
            
            if no2_trop.ndim == 3:
                no2_trop = no2_trop[0]
                qa_flag = qa_flag[0]
            elif no2_trop.ndim == 1:
                if lat.ndim == 2 and lon.ndim == 2:
                    grid_shape = lat.shape
                else:
                    n_lat = len(np.unique(lat))
                    n_lon = len(np.unique(lon))
                    expected_size = n_lat * n_lon
                    
                    if len(no2_trop) == expected_size:
                        grid_shape = (n_lat, n_lon)
                    else:
                        ds_product.close()
                        return None
                
                try:
                    no2_trop = no2_trop.reshape(grid_shape)
                    qa_flag = qa_flag.reshape(grid_shape)
                except:
                    ds_product.close()
                    return None
            
            ds_product.close()
            
            ds_support = xr.open_dataset(filepath, group='support_data')
            cloud_frac = ds_support['eff_cloud_fraction'].values
            
            if cloud_frac.ndim == 3:
                cloud_frac = cloud_frac[0]
            elif cloud_frac.ndim == 1:
                try:
                    cloud_frac = cloud_frac.reshape(no2_trop.shape)
                except:
                    cloud_frac = np.zeros_like(no2_trop)
            
            ds_support.close()
            
            if lat.ndim == 2:
                lat_unique = np.unique(lat)
                lon_unique = np.unique(lon)
            else:
                lat_unique = lat
                lon_unique = lon
            
            lat_idx = np.where((lat_unique >= self.lat_min) & (lat_unique <= self.lat_max))[0]
            lon_idx = np.where((lon_unique >= self.lon_min) & (lon_unique <= self.lon_max))[0]
            
            if len(lat_idx) == 0 or len(lon_idx) == 0:
                return None
            
            if lat.ndim == 2:
                lat_mask = (lat >= self.lat_min) & (lat <= self.lat_max)
                lon_mask = (lon >= self.lon_min) & (lon <= self.lon_max)
                combined_mask = lat_mask & lon_mask
                
                rows, cols = np.where(combined_mask)
                if len(rows) == 0:
                    return None
                
                row_min, row_max = rows.min(), rows.max()
                col_min, col_max = cols.min(), cols.max()
                
                la_no2 = no2_trop[row_min:row_max+1, col_min:col_max+1]
                la_qa = qa_flag[row_min:row_max+1, col_min:col_max+1]
                la_cloud = cloud_frac[row_min:row_max+1, col_min:col_max+1]
            else:
                la_no2 = no2_trop[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
                la_qa = qa_flag[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
                la_cloud = cloud_frac[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
            
            la_no2_filtered = np.where(
                (la_qa == 0) & (la_cloud < 0.5),
                la_no2,
                np.nan
            )
            
            if np.all(np.isnan(la_no2_filtered)):
                return None
            
            if np.any(np.isnan(la_no2_filtered)):
                valid_mean = np.nanmean(la_no2_filtered)
                if np.isnan(valid_mean):
                    return None
                la_no2_filtered = np.where(np.isnan(la_no2_filtered), valid_mean, la_no2_filtered)
            
            zoom_factors = (self.dim / la_no2_filtered.shape[0], 
                           self.dim / la_no2_filtered.shape[1])
            
            resampled = zoom(la_no2_filtered, zoom_factors, order=1)
            
            if resampled.shape != (self.dim, self.dim):
                result = np.zeros((self.dim, self.dim), dtype=np.float32)
                min_r = min(self.dim, resampled.shape[0])
                min_c = min(self.dim, resampled.shape[1])
                result[:min_r, :min_c] = resampled[:min_r, :min_c]
                resampled = result
            
            resampled = np.clip(resampled / 1e16, 0, 10)
            
            return {
                'array': resampled.astype(np.float32),
                'timestamp': timestamp
            }
            
        except Exception as e:
            return None
    
    def _save_hourly_data(self):
        print(f"\nSaving continuous hourly data...")
        print(f"   Total hours collected: {len(self.continuous_hourly_data)}")
        print(f"   Expected hours: {len(self.all_timestamps)}")
        
        all_data = np.stack(self.continuous_hourly_data, axis=0)
        
        non_min = np.sum(all_data > 0.001)
        total = all_data.size
        coverage = non_min / total * 100
        
        hourly_means = []
        for h in range(24):
            hour_mask = [ts.hour == h for ts in self.continuous_timestamps]
            if any(hour_mask):
                hour_data = all_data[hour_mask]
                hourly_means.append(np.mean(hour_data))
            else:
                hourly_means.append(0)
        
        print(f"\nData Statistics:")
        print(f"   Shape: {all_data.shape}")
        print(f"   Min value: {np.min(all_data):.6f}")
        print(f"   Max value: {np.max(all_data):.6f}")
        print(f"   Mean value: {np.mean(all_data):.6f}")
        print(f"   Std dev: {np.std(all_data):.6f}")
        print(f"   Data coverage: {coverage:.2f}%")
        
        print(f"\nDiurnal Pattern (PST):")
        for h in range(24):
            pst_hour = (h - 8) % 24
            print(f"   {pst_hour:02d}:00 - Mean: {hourly_means[h]:.6f}")
        
        date_suffix = f"_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}"
        if self.test_mode:
            date_suffix = f"_TEST{date_suffix}"
        
        output_file = os.path.join(self.output_dir, 
                                  f"tempo_l3_no2{date_suffix}_hourly.npz")
        
        metadata = {
            'start_date': str(self.start_date),
            'end_date': str(self.end_date),
            'extent': self.extent,
            'dim': self.dim,
            'n_hours': len(all_data),
            'shape': all_data.shape,
            'filling_strategy': self.filling_strategy,
            'decay_hours': self.decay_hours,
            'test_mode': self.test_mode
        }
        
        np.savez_compressed(
            output_file,
            data=all_data,
            timestamps=self.continuous_timestamps,
            hourly_means=hourly_means,
            metadata=metadata
        )
        
        print(f"\nSaved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024**3):.2f} GB")

def main():
    
    cache_dir = 'data/tempo_l3_cache/'
    output_dir = 'data/tempo_l3_processed/'
    
    try:
        with open('/proc/version', 'r') as f:
            if 'microsoft' in f.read().lower():
                d_stat = os.statvfs('/mnt/d') if os.path.exists('/mnt/d') else None
                c_stat = os.statvfs('/mnt/c') if os.path.exists('/mnt/c') else None
                
                if d_stat and (not c_stat or d_stat.f_bavail > c_stat.f_bavail):
                    cache_dir = '/mnt/d/tempo_cache_fixed/'
                    print("WSL detected - using D: drive for TEMPO downloads")
                elif c_stat:
                    cache_dir = '/mnt/c/tempo_cache/'
                    print("WSL detected - using C: drive for TEMPO downloads")
                
                print("Processed files will stay in WSL (they're small)")
    except:
        pass
    
    config = {
        'start_date': '2023-08-02',
        'end_date': '2025-08-02',
        'extent': (-118.75, -117.5, 33.5, 34.5),
        'filling_strategy': 'persistence',
        'dim': 40,
        'cache_dir': cache_dir,
        'output_dir': output_dir,
        'max_retries': 10,
        'initial_wait': 5,
        'verbose': True,
        'test_mode': False
    }
    
    processor = TempoL3Processor(**config)
    
    processor.process_all_months()
    
    print("\n🎉 Processing complete!")
    print(f"Generated exactly {len(processor.continuous_hourly_data)} hours of data")
    print(f"This should match AirNow's data length for ConvLSTM alignment")
    
    if config['test_mode']:
        print("\nTEST MODE RESULTS:")
        print("   - Processed 1 week of data")
        print("   - Set test_mode=False for full dataset")

if __name__ == "__main__":
    main()