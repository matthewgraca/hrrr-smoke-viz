import os
import requests
import netrc
import earthaccess
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
from scipy.ndimage import zoom
import time


class TempoNO2Data:
    def __init__(
        self,
        start_date="2023-08-02 00:00",
        end_date="2025-08-02 00:00",
        extent=(-118.615, -117.70, 33.60, 34.35),
        dim=84,
        raw_dir='data/tempo_v03_raw/',
        processed_dir='data/tempo_v03_processed/',
        n_threads=4,
        max_retries=5,
        initial_wait=2,
        cloud_threshold=0.5,
        test_mode=False
    ):
        if test_mode:
            self.start_date = pd.to_datetime(start_date)
            self.end_date = self.start_date + timedelta(days=7)
            print("TEST MODE: Running for 1 week only")
        else:
            self.start_date = pd.to_datetime(start_date)
            self.end_date = pd.to_datetime(end_date)
        
        self.extent = extent
        self.lon_min, self.lon_max, self.lat_min, self.lat_max = extent
        self.dim = dim
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.n_threads = n_threads
        self.max_retries = max_retries
        self.initial_wait = initial_wait
        self.cloud_threshold = cloud_threshold
        self.test_mode = test_mode
        
        self.collection_id = 'C2930763263-LARC_CLOUD'
        self.harmony_root = 'https://harmony.earthdata.nasa.gov'
        
        self.variables = [
            'product/vertical_column_troposphere',
            'product/main_data_quality_flag',
            'support_data/eff_cloud_fraction'
        ]
        
        self.all_timestamps = pd.date_range(self.start_date, self.end_date, freq='h', inclusive='left')
        
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        
        self.bearer_token = self._setup_auth()
        
        print("TEMPO NO2 L3 V03 Data Pipeline")
        print("=" * 70)
        print(f"Date range: {self.start_date} to {self.end_date}")
        print(f"Total hours: {len(self.all_timestamps)}")
        print(f"LA extent: {self.extent}")
        print(f"Output grid: {self.dim}x{self.dim}")
        print(f"Cloud threshold: {self.cloud_threshold}")
        print("=" * 70)
    
    def _setup_auth(self):
        earthaccess.login()
        username, _, password = netrc.netrc().authenticators('urs.earthdata.nasa.gov')
        
        auth_response = requests.get(
            'https://urs.earthdata.nasa.gov/api/users/tokens',
            auth=(username, password)
        )
        tokens = auth_response.json()
        
        if tokens:
            return tokens[0]['access_token']
        else:
            token_response = requests.post(
                'https://urs.earthdata.nasa.gov/api/users/token',
                auth=(username, password)
            )
            return token_response.json()['access_token']
    
    def _submit_harmony_job(self, start_time, end_time):
        var_string = ','.join(quote(v, safe='') for v in self.variables)
        time_start = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        time_end = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        url = (
            f"{self.harmony_root}/{self.collection_id}/ogc-api-coverages/1.0.0/"
            f"collections/{var_string}/coverage/rangeset"
            f"?subset=time(\"{time_start}\":\"{time_end}\")"
            f"&subset=lat({self.lat_min}:{self.lat_max})"
            f"&subset=lon({self.lon_min}:{self.lon_max})"
            f"&maxResults=500"
            f"&format=application/x-netcdf4"
        )
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    headers={'Authorization': f'Bearer {self.bearer_token}'},
                    allow_redirects=True,
                    timeout=60
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"   Attempt {attempt+1}: HTTP {response.status_code}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.initial_wait * (2 ** attempt))
            except Exception as e:
                print(f"   Attempt {attempt+1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.initial_wait * (2 ** attempt))
        return None
    
    def _wait_for_job(self, job_id):
        job_url = f"{self.harmony_root}/jobs/{job_id}"
        
        while True:
            try:
                response = requests.get(
                    job_url,
                    headers={'Authorization': f'Bearer {self.bearer_token}'},
                    timeout=30
                )
                job_info = response.json()
                status = job_info.get('status')
                progress = job_info.get('progress', 0)
                
                if status == 'successful':
                    return job_info
                elif status == 'failed':
                    print(f"\n   Job failed: {job_info.get('message', 'Unknown error')}")
                    return None
                elif status == 'paused':
                    print(f"\n   Job paused, resuming...")
                    resume_url = f"{self.harmony_root}/jobs/{job_id}/resume"
                    requests.post(resume_url, headers={'Authorization': f'Bearer {self.bearer_token}'})
                    time.sleep(5)
                elif status in ['running', 'previewing', 'running_with_errors']:
                    print(f"   Processing: {progress}%", end='\r')
                    time.sleep(5)
                else:
                    print(f"\n   Unknown status: {status}, waiting...")
                    time.sleep(10)
            except Exception as e:
                print(f"\n   Error checking job: {e}")
                time.sleep(10)
    
    def _download_file(self, url, output_path):
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    headers={'Authorization': f'Bearer {self.bearer_token}'},
                    stream=True,
                    timeout=120
                )
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    return True
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.initial_wait * (2 ** attempt))
        return False
    
    def _download_raw(self):
        print("\n" + "=" * 70)
        print("DOWNLOADING RAW DATA")
        print("=" * 70)
        
        total_downloaded = 0
        current = self.start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_limit = self.end_date
        
        while current <= end_limit:
            year, month = current.year, current.month
            
            # Determine if this is the current/end month (partial month)
            is_end_month = (year == end_limit.year and month == end_limit.month)
            is_start_month = (year == self.start_date.year and month == self.start_date.month)
            
            # Calculate month boundaries
            if month == 12:
                next_month = datetime(year+1, 1, 1)
            else:
                next_month = datetime(year, month+1, 1)
            
            # Set actual request range (preserving hour for start/end)
            if is_start_month:
                request_start = self.start_date.to_pydatetime()
            else:
                request_start = datetime(year, month, 1)
            
            if is_end_month:
                request_end = end_limit.to_pydatetime()
            else:
                request_end = next_month - timedelta(seconds=1)
            
            if request_end < request_start:
                current = next_month
                continue
            
            # For partial months (start or end), use datetime-based directory naming
            if is_start_month or is_end_month:
                month_dir = os.path.join(
                    self.raw_dir, 
                    f"{year}", 
                    f"{month:02d}",
                    f"{request_start.strftime('%Y%m%d%H')}_{request_end.strftime('%Y%m%d%H')}"
                )
            else:
                month_dir = os.path.join(self.raw_dir, f"{year}", f"{month:02d}")
            
            # Check for existing files
            if os.path.exists(month_dir):
                existing = [f for f in os.listdir(month_dir) if f.endswith('.nc4') or f.endswith('.nc')]
                if len(existing) > 0:
                    print(f"\n{year}-{month:02d}: Found {len(existing)} files in {month_dir}, skipping...")
                    total_downloaded += len(existing)
                    current = next_month
                    continue
            
            os.makedirs(month_dir, exist_ok=True)
            
            date_range_str = f"{request_start.strftime('%Y-%m-%d %H:%M')} to {request_end.strftime('%Y-%m-%d %H:%M')}"
            print(f"\n{year}-{month:02d}: Submitting Harmony job for {date_range_str}...")
            
            job_response = self._submit_harmony_job(request_start, request_end)
            if not job_response:
                print(f"   Failed to submit job, skipping")
                current = next_month
                continue
            
            job_id = job_response.get('jobID')
            if not job_id:
                print(f"   No job ID returned, skipping")
                current = next_month
                continue
            
            print(f"   Job ID: {job_id}")
            job_result = self._wait_for_job(job_id)
            
            if not job_result:
                current = next_month
                continue
            
            links = [l for l in job_result.get('links', []) if l.get('rel') == 'data']
            print(f"\n   Downloading {len(links)} files...")
            
            downloaded = 0
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                futures = {}
                for link in links:
                    url = link['href']
                    filename = link.get('title', url.split('/')[-1])
                    output_path = os.path.join(month_dir, filename)
                    futures[executor.submit(self._download_file, url, output_path)] = filename
                
                for future in as_completed(futures):
                    if future.result():
                        downloaded += 1
                        if downloaded % 20 == 0:
                            print(f"   Downloaded: {downloaded}/{len(links)}")
            
            print(f"   Done: {downloaded} files")
            total_downloaded += downloaded
            current = next_month
        
        print(f"\nDownload complete! Total files: {total_downloaded}")
        return total_downloaded
    
    def _get_timestamp_from_filename(self, filename):
        parts = filename.split('_')
        ts_str = parts[4]
        return pd.to_datetime(ts_str, format='%Y%m%dT%H%M%SZ')
    
    def _load_raw_file(self, filepath):
        try:
            ds_prod = xr.open_dataset(filepath, group='product')
            no2 = ds_prod['vertical_column_troposphere'].values.squeeze()
            qa_flag = ds_prod['main_data_quality_flag'].values.squeeze()
            ds_prod.close()
            
            ds_sup = xr.open_dataset(filepath, group='support_data')
            cloud_frac = ds_sup['eff_cloud_fraction'].values.squeeze()
            ds_sup.close()
            
            no2_filtered = np.where(
                (qa_flag <= 0) & (cloud_frac < self.cloud_threshold),
                no2,
                np.nan
            )
            
            valid_frac = np.sum(~np.isnan(no2_filtered)) / no2_filtered.size
            if valid_frac < 0.1:
                return None
            
            valid_mean = np.nanmean(no2_filtered)
            no2_filled = np.where(np.isnan(no2_filtered), valid_mean, no2_filtered)
            
            zoom_factors = (self.dim / no2_filled.shape[0], self.dim / no2_filled.shape[1])
            resampled = zoom(no2_filled, zoom_factors, order=1)
            
            if resampled.shape != (self.dim, self.dim):
                result = np.zeros((self.dim, self.dim), dtype=np.float32)
                min_r = min(self.dim, resampled.shape[0])
                min_c = min(self.dim, resampled.shape[1])
                result[:min_r, :min_c] = resampled[:min_r, :min_c]
                resampled = result
            
            resampled = np.flip(resampled, axis=0)
            resampled = (resampled / 1e16).astype(np.float32)
            return resampled
        except Exception as e:
            print(f"   Error loading {filepath}: {e}")
            return None
    
    def _process(self):
        print("\nProcessing QA=0 (strict)...")
        
        all_files = []
        for root, dirs, files in os.walk(self.raw_dir):
            for f in files:
                if f.endswith('.nc4') or f.endswith('.nc'):
                    filepath = os.path.join(root, f)
                    ts = self._get_timestamp_from_filename(f)
                    if self.start_date <= ts < self.end_date:
                        all_files.append((ts, filepath))
        
        all_files = sorted(all_files, key=lambda x: x[0])
        print(f"Found {len(all_files)} raw files in date range")
        
        file_by_hour = {}
        for ts, filepath in all_files:
            hour_key = ts.floor('h')
            if hour_key not in file_by_hour or ts > file_by_hour[hour_key][0]:
                file_by_hour[hour_key] = (ts, filepath)
        
        print(f"Unique hours with data: {len(file_by_hour)}")
        
        hourly_data = []
        last_valid = None
        valid_count = 0
        filled_count = 0
        fill_log = []
        
        for i, hour in enumerate(self.all_timestamps):
            if (i + 1) % 500 == 0:
                print(f"   Processing hour {i+1}/{len(self.all_timestamps)}")
            
            if hour in file_by_hour:
                ts, filepath = file_by_hour[hour]
                data = self._load_raw_file(filepath)
                
                if data is not None:
                    hourly_data.append(data)
                    last_valid = data.copy()
                    valid_count += 1
                    fill_log.append((hour, 'valid', ts))
                else:
                    if last_valid is not None:
                        hourly_data.append(last_valid.copy())
                        fill_log.append((hour, 'forward_fill_bad_qa', ts))
                    else:
                        hourly_data.append(np.zeros((self.dim, self.dim), dtype=np.float32))
                        fill_log.append((hour, 'zero_no_prior', ts))
                    filled_count += 1
            else:
                if last_valid is not None:
                    hourly_data.append(last_valid.copy())
                    fill_log.append((hour, 'forward_fill_missing', None))
                else:
                    hourly_data.append(np.zeros((self.dim, self.dim), dtype=np.float32))
                    fill_log.append((hour, 'zero_no_prior', None))
                filled_count += 1
        
        print(f"   Valid hours: {valid_count}")
        print(f"   Forward-filled hours: {filled_count}")
        
        if self.test_mode:
            self._verify_forward_fill(fill_log, hourly_data)
        
        data_array = np.stack(hourly_data, axis=0)
        
        output_file = os.path.join(
            self.processed_dir,
            f"tempo_no2_la_hourly_qa0_{self.start_date.strftime('%Y%m%d%H')}_{self.end_date.strftime('%Y%m%d%H')}.npz"
        )
        
        print(f"\n   Saving...")
        print(f"   Shape: {data_array.shape}")
        print(f"   Min: {np.nanmin(data_array):.6f}")
        print(f"   Max: {np.nanmax(data_array):.6f}")
        print(f"   Mean: {np.nanmean(data_array):.6f}")
        
        np.savez_compressed(
            output_file,
            data=data_array,
            timestamps=[str(ts) for ts in self.all_timestamps],
            extent=self.extent,
            dim=self.dim,
            qa_threshold=0,
            cloud_threshold=self.cloud_threshold,
            valid_count=valid_count,
            filled_count=filled_count
        )
        
        size_mb = os.path.getsize(output_file) / (1024**2)
        print(f"   Saved to: {output_file}")
        print(f"   File size: {size_mb:.2f} MB")
        
        return data_array
    
    def _verify_forward_fill(self, fill_log, hourly_data):
        print("\n" + "=" * 70)
        print("FORWARD FILL VERIFICATION")
        print("=" * 70)
        
        status_counts = {}
        for hour, status, src_ts in fill_log:
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("\nStatus breakdown:")
        for status, count in sorted(status_counts.items()):
            pct = 100 * count / len(fill_log)
            print(f"   {status}: {count} ({pct:.1f}%)")
        
        print("\nHour-by-hour log:")
        print("-" * 70)
        for i, (hour, status, src_ts) in enumerate(fill_log):
            src_str = f"from {src_ts}" if src_ts else ""
            print(f"   {hour} -> {status} {src_str}")
        
        print("\nVerifying forward fill correctness...")
        errors = []
        last_valid_idx = None
        
        for i, (hour, status, src_ts) in enumerate(fill_log):
            if status == 'valid':
                last_valid_idx = i
            elif status in ['forward_fill_missing', 'forward_fill_bad_qa']:
                if last_valid_idx is not None:
                    if not np.allclose(hourly_data[i], hourly_data[last_valid_idx]):
                        errors.append(f"Hour {i} ({hour}): forward fill mismatch with last valid at {last_valid_idx}")
            elif status == 'zero_no_prior':
                if not np.allclose(hourly_data[i], 0):
                    errors.append(f"Hour {i} ({hour}): expected zeros but got non-zero data")
        
        if errors:
            print("\nERRORS FOUND:")
            for e in errors:
                print(f"   {e}")
        else:
            print("\nAll forward fills verified correctly!")
        
        print("-" * 70)
    
    def run(self):
        start_time = time.time()
        self._download_raw()
        
        print("\n" + "=" * 70)
        print("PROCESSING TO HOURLY ALIGNED GRID")
        print("=" * 70)
        
        self._process()
        
        total_time = time.time() - start_time
        hours = total_time / 3600
        
        print(f"\n{'=' * 70}")
        print(f"COMPLETE!")
        print(f"Total time: {hours:.2f} hours ({total_time:.0f} seconds)")
        print(f"{'=' * 70}")


def main():
    processor = TempoNO2Data(
        start_date='2023-08-02 00:00',
        end_date='2025-08-02 00:00',
        extent=(-118.615, -117.70, 33.60, 34.35),
        dim=84,
        raw_dir='data/tempo_v03_raw_test/',
        processed_dir='data/tempo_v03_processed/',
        n_threads=4,
        max_retries=5,
        cloud_threshold=0.5,
        test_mode=False
    )
    processor.run()


if __name__ == "__main__":
    main()