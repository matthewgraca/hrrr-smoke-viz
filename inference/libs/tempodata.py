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
import time, re


class TempoNO2Data:
    def __init__(
        self,
        start_date="2023-08-02 00:00",
        end_date="2025-08-02 00:00",
        extent=(-118.615, -117.70, 33.60, 34.35),
        dim=84,
        raw_dir='data/tempo_nrt_raw/',
        processed_dir='data/tempo_nrt_processed/',
        n_threads=4,
        max_retries=5,
        initial_wait=2,
        cloud_threshold=0.5,
        use_nrt=True,
        expected_latency_hours=4,  # NRT should be ~3h, buffer to 4
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
        self.use_nrt = use_nrt
        self.expected_latency_hours = expected_latency_hours
        self.test_mode = test_mode
        
        # NRT L3 vs Standard L3
        if use_nrt:
            self.collection_id = 'C3685668637-LARC_CLOUD'  # TEMPO_NO2_L3_NRT
            self.short_name = 'TEMPO_NO2_L3_NRT'
        else:
            self.collection_id = 'C3685896708-LARC_CLOUD'  # TEMPO_NO2_L3 V04
            self.short_name = 'TEMPO_NO2_L3'
        
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
        
        print(f"TEMPO NO2 L3 {'NRT' if use_nrt else 'Standard'} Data Pipeline")
        print("=" * 70)
        print(f"Collection: {self.short_name}")
        print(f"Date range: {self.start_date} to {self.end_date}")
        print(f"Total hours: {len(self.all_timestamps)}")
        print(f"LA extent: {self.extent}")
        print(f"Output grid: {self.dim}x{self.dim}")
        print(f"Cloud threshold: {self.cloud_threshold}")
        print(f"Expected latency: {self.expected_latency_hours}h (will forward-fill)")
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
    
    def _check_data_availability(self):
            """Check latest available data and report latency."""
            print("\nChecking data availability...")
            
            now = datetime.utcnow()
            # FIX: earthaccess returns oldest-first, so 14 days + count=50 only returned 
            # old granules and missed recent data. Narrowed to 3 days to ensure we get latest while keeping enough of a buffer to make sure there is redundancy.
            search_start = now - timedelta(days=3)
            
            try:
                granules = earthaccess.search_data(
                    short_name=self.short_name,
                    temporal=(search_start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")),
                    count=100
                )
                
                if not granules:
                    print("   WARNING: No granules found in last 3 days!")
                    return None, None
                
                granule_times = []
                for g in granules:
                    try:
                        umm = g['umm']
                        temporal = umm.get('TemporalExtent', {})
                        range_dt = temporal.get('RangeDateTime', {})
                        end_time = range_dt.get('EndingDateTime')
                        if end_time:
                            granule_times.append(pd.to_datetime(end_time))
                    except (KeyError, TypeError):
                        continue
                
                if not granule_times:
                    print("   WARNING: Could not parse any granule timestamps!")
                    return None, None
                
                latest = max(granule_times).to_pydatetime().replace(tzinfo=None)
                latency_hours = (now - latest).total_seconds() / 3600
                
                print(f"   Latest data: {latest.strftime('%Y-%m-%d %H:%M')} UTC")
                print(f"   Current time: {now.strftime('%Y-%m-%d %H:%M')} UTC")
                print(f"   Latency: {latency_hours:.1f} hours")
                
                if latency_hours > 24:
                    print(f"   ⚠️  WARNING: Data latency ({latency_hours:.1f}h) exceeds 24 hours!")
                    print(f"   This may indicate a data outage. Will forward-fill from last available.")
                
                return latest, latency_hours
                
            except Exception as e:
                print(f"   Error checking availability: {e}")
                import traceback
                traceback.print_exc()
                return None, None
    
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
        
        # Check current data availability first
        latest_available, current_latency = self._check_data_availability()
        
        total_downloaded = 0
        current = self.start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_limit = self.end_date
        
        while current <= end_limit:
            year, month = current.year, current.month
            
            is_end_month = (year == end_limit.year and month == end_limit.month)
            is_start_month = (year == self.start_date.year and month == self.start_date.month)
            
            if month == 12:
                next_month = datetime(year+1, 1, 1)
            else:
                next_month = datetime(year, month+1, 1)
            
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
            
            if is_start_month or is_end_month:
                month_dir = os.path.join(
                    self.raw_dir, 
                    f"{year}", 
                    f"{month:02d}",
                    f"{request_start.strftime('%Y%m%d%H')}_{request_end.strftime('%Y%m%d%H')}"
                )
            else:
                month_dir = os.path.join(self.raw_dir, f"{year}", f"{month:02d}")
            
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
        return total_downloaded, latest_available
    
    def _get_timestamp_from_filename(self, filename):
        match = re.search(r'(\d{8}T\d{6}Z)', filename)
        if match:
            return pd.to_datetime(match.group(1), format='%Y%m%dT%H%M%SZ').tz_localize(None)
        raise ValueError(f"Could not find timestamp in filename: {filename}")
    
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
    
    def _find_last_available_data(self, file_by_hour, target_hour, max_lookback_days=30):
        """
        Search backwards from target_hour to find the most recent valid data.
        Returns (hour_key, filepath) or (None, None) if nothing found.
        """
        lookback_hours = max_lookback_days * 24
        
        for i in range(lookback_hours):
            check_hour = target_hour - timedelta(hours=i)
            if check_hour in file_by_hour:
                return check_hour, file_by_hour[check_hour]
        
        return None, None
    
    def _process(self, latest_available=None):
        print("\nProcessing with forward-fill strategy...")
        
        all_files = []
        for root, dirs, files in os.walk(self.raw_dir):
            for f in files:
                if f.endswith('.nc4') or f.endswith('.nc'):
                    filepath = os.path.join(root, f)
                    try:
                        ts = self._get_timestamp_from_filename(f)
                        if self.start_date <= ts < self.end_date:
                            all_files.append((ts, filepath))
                    except:
                        continue
        
        all_files = sorted(all_files, key=lambda x: x[0])
        print(f"Found {len(all_files)} raw files in date range")
        
        # Map files to hours
        file_by_hour = {}
        for ts, filepath in all_files:
            hour_key = ts.floor('h')
            if hour_key not in file_by_hour or ts > file_by_hour[hour_key][0]:
                file_by_hour[hour_key] = (ts, filepath)
        
        print(f"Unique hours with data: {len(file_by_hour)}")
        
        # Determine effective end time (latest available or requested end)
        now = pd.Timestamp.utcnow().floor('h').tz_localize(None)
        effective_end = min(self.end_date, now)
        
        if latest_available:
            latest_hour = pd.Timestamp(latest_available).floor('h')
            hours_missing = (effective_end - latest_hour).total_seconds() / 3600
            if hours_missing > 0:
                print(f"\n⚠️  Data gap detected: {hours_missing:.0f} hours will be forward-filled")
                print(f"   Last available: {latest_hour}")
                print(f"   Requested end: {effective_end}")
        
        # Process all hours with forward-fill
        hourly_data = []
        last_valid_data = None
        last_valid_hour = None
        valid_count = 0
        forward_fill_count = 0
        gap_fill_count = 0
        fill_log = []
        
        for i, hour in enumerate(self.all_timestamps):
            if (i + 1) % 500 == 0:
                print(f"   Processing hour {i+1}/{len(self.all_timestamps)}")
            
            if hour in file_by_hour:
                ts, filepath = file_by_hour[hour]
                data = self._load_raw_file(filepath)
                
                if data is not None:
                    hourly_data.append(data)
                    last_valid_data = data.copy()
                    last_valid_hour = hour
                    valid_count += 1
                    fill_log.append((hour, 'valid', ts))
                else:
                    # File exists but QA failed - forward fill
                    if last_valid_data is not None:
                        hourly_data.append(last_valid_data.copy())
                        fill_log.append((hour, 'forward_fill_bad_qa', f"from {last_valid_hour}"))
                        forward_fill_count += 1
                    else:
                        # No prior data - search backwards
                        found_hour, found_file = self._find_last_available_data(file_by_hour, hour)
                        if found_file:
                            ts_found, fp_found = found_file
                            data = self._load_raw_file(fp_found)
                            if data is not None:
                                hourly_data.append(data)
                                last_valid_data = data.copy()
                                last_valid_hour = found_hour
                                fill_log.append((hour, 'backfill_found', f"from {found_hour}"))
                                gap_fill_count += 1
                            else:
                                hourly_data.append(np.zeros((self.dim, self.dim), dtype=np.float32))
                                fill_log.append((hour, 'zero_no_valid', None))
                        else:
                            hourly_data.append(np.zeros((self.dim, self.dim), dtype=np.float32))
                            fill_log.append((hour, 'zero_no_prior', None))
            else:
                # No file for this hour - forward fill from last valid
                if last_valid_data is not None:
                    hourly_data.append(last_valid_data.copy())
                    fill_log.append((hour, 'forward_fill_missing', f"from {last_valid_hour}"))
                    forward_fill_count += 1
                else:
                    # No prior data yet - search backwards
                    found_hour, found_file = self._find_last_available_data(file_by_hour, hour)
                    if found_file:
                        ts_found, fp_found = found_file
                        data = self._load_raw_file(fp_found)
                        if data is not None:
                            hourly_data.append(data)
                            last_valid_data = data.copy()
                            last_valid_hour = found_hour
                            fill_log.append((hour, 'backfill_found', f"from {found_hour}"))
                            gap_fill_count += 1
                        else:
                            hourly_data.append(np.zeros((self.dim, self.dim), dtype=np.float32))
                            fill_log.append((hour, 'zero_no_valid', None))
                    else:
                        hourly_data.append(np.zeros((self.dim, self.dim), dtype=np.float32))
                        fill_log.append((hour, 'zero_no_prior', None))
        
        # Summary
        print(f"\n   Processing summary:")
        print(f"   Valid hours: {valid_count}")
        print(f"   Forward-filled: {forward_fill_count}")
        print(f"   Gap backfills: {gap_fill_count}")
        print(f"   Zero-filled (no data): {len(self.all_timestamps) - valid_count - forward_fill_count - gap_fill_count}")
        
        if self.test_mode:
            self._verify_forward_fill(fill_log, hourly_data)
        
        data_array = np.stack(hourly_data, axis=0)
        
        nrt_suffix = "_nrt" if self.use_nrt else ""
        output_file = os.path.join(
            self.processed_dir,
            f"tempo_no2_la_hourly{nrt_suffix}_{self.start_date.strftime('%Y%m%d%H')}_{self.end_date.strftime('%Y%m%d%H')}.npz"
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
            forward_fill_count=forward_fill_count,
            gap_fill_count=gap_fill_count,
            use_nrt=self.use_nrt,
            last_valid_hour=str(last_valid_hour) if last_valid_hour else None
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
        for hour, status, src in fill_log:
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("\nStatus breakdown:")
        for status, count in sorted(status_counts.items()):
            pct = 100 * count / len(fill_log)
            print(f"   {status}: {count} ({pct:.1f}%)")
        
        print("\nFirst 50 hours:")
        print("-" * 70)
        for i, (hour, status, src) in enumerate(fill_log[:50]):
            src_str = f" {src}" if src else ""
            print(f"   {hour} -> {status}{src_str}")
        
        print("-" * 70)
    
    def run(self):
        start_time = time.time()
        total_downloaded, latest_available = self._download_raw()
        
        print("\n" + "=" * 70)
        print("PROCESSING TO HOURLY ALIGNED GRID")
        print("=" * 70)
        
        self._process(latest_available)
        
        total_time = time.time() - start_time
        hours = total_time / 3600
        
        print(f"\n{'=' * 70}")
        print(f"COMPLETE!")
        print(f"Total time: {hours:.2f} hours ({total_time:.0f} seconds)")
        print(f"{'=' * 70}")


def main():
    processor = TempoNO2Data(
        start_date='2023-09-17 00:00',  # NRT V02 starts Sept 17, 2025
        end_date='2026-02-02 18:00',     # Current time
        extent=(-118.615, -117.70, 33.60, 34.35),
        dim=84,
        raw_dir='data/tempo_nrt_raw/',
        processed_dir='data/tempo_nrt_processed/',
        n_threads=4,
        max_retries=5,
        cloud_threshold=0.5,
        use_nrt=True,  # Use NRT L3 for best latency
        expected_latency_hours=4,
        test_mode=False
    )
    processor.run()


if __name__ == "__main__":
    main()