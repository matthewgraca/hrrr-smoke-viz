import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from herbie import Herbie
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import ConnectionError
from scipy.interpolate import griddata
from tqdm import tqdm
import gc


class HRRRData:
    DEFAULT_VARIABLES = {
        'temp_2m': {'search': ':TMP:2 m above ground:', 'var_names': ['t2m']},
        'pbl_height': {'search': ':HPBL:surface:', 'var_names': ['blh']},
         # 'precip_rate': {'search': ':PRATE:surface:', 'var_names': ['prate']},
        'smoke_massden': {'search': ':MASSDEN:8 m above ground:', 'var_names': ['unknown'], 'grib_name': 'Mass density', 'unit_scale': 1e9},
        'u_wind': {'search': ':UGRD:10 m above ground:', 'var_names': ['u10']},
        'v_wind': {'search': ':VGRD:10 m above ground:', 'var_names': ['v10']},
    }

    HRRR_LON_0 = -97.5
    HRRR_LAT_REF = 38.5

    SPECIAL_RUN_HOURS = [0, 6, 12, 18]

    def __init__(
        self,
        start_date="2023-08-02-00",
        end_date="2023-08-03-00",
        extent=None,
        extent_name='la_basin',
        output_dir='data/hrrr',
        grid_size=84,
        chunk_months=2,
        force_reprocess=False,
        verbose=False,
        max_threads=20,
        forecast=False,
        historical_forecast=False,
        variables=None,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.extent = extent
        self.extent_name = extent_name
        self.output_dir = output_dir
        self.chunk_months = chunk_months
        self.verbose = verbose
        self.max_threads = max_threads
        self.force_reprocess = force_reprocess
        self.grid_size = grid_size
        self.forecast = forecast
        self.historical_forecast = historical_forecast

        self.lon_regular = np.linspace(extent[0], extent[1], self.grid_size)
        self.lat_regular = np.linspace(extent[2], extent[3], self.grid_size)
        self.lon_grid, self.lat_grid = np.meshgrid(self.lon_regular, self.lat_regular)

        self.variables = variables if variables is not None else self.DEFAULT_VARIABLES
        self.combined_search = '|'.join([v['search'].strip(':') for v in self.variables.values()])

        self._validate_variables()

        os.makedirs(output_dir, exist_ok=True)

        if historical_forecast:
            self.run_historical_forecast_mode()
        elif forecast:
            self.run_forecast_mode()
        else:
            self.run_observed_mode()

    def _resolve_var(self, ds, var_info):
        """Find the xarray variable name for a given variable config."""
        for possible_name in var_info['var_names']:
            if possible_name in ds:
                return possible_name

        grib_name = var_info.get('grib_name')
        if grib_name:
            for ds_var in ds.data_vars:
                if ds[ds_var].attrs.get('GRIB_parameterName') == grib_name:
                    return ds_var

        return None

    def _rotate_wind_to_earth(self, u_grid, v_grid, lats, lons):
        """Rotate HRRR grid-relative U/V to earth-relative.
        
        HRRR uses Lambert Conformal projection where U/V wind components
        are aligned to grid axes, not geographic east/north. This applies
        the rotation angle correction (~13° over LA Basin).
        """
        lons_deg = lons if lons.min() < 0 else lons - 360
        alpha = np.arctan(
            np.tan(np.radians(lons_deg - self.HRRR_LON_0)) * np.sin(np.radians(self.HRRR_LAT_REF))
        )
        u_earth = u_grid * np.cos(alpha) + v_grid * np.sin(alpha)
        v_earth = -u_grid * np.sin(alpha) + v_grid * np.cos(alpha)
        return u_earth, v_earth

    def _extract_and_interpolate(self, ds):
        """Extract all variables from a dataset, rotate wind, and interpolate to regular grid.
        
        Handles the full pipeline: subset on native grid, rotate U/V wind
        to earth-relative, then interpolate everything to the 84x84 lat/lon grid.
        """
        subsets = {}
        for var_name, var_info in self.variables.items():
            found_var = self._resolve_var(ds, var_info)
            if found_var is None:
                raise ValueError(
                    f"Variable '{var_name}' not found in dataset. "
                    f"Available: {list(ds.data_vars)}"
                )
            data_subset, lats_subset, lons_subset = self.subset_and_get_coords(ds, found_var)
            scale = var_info.get('unit_scale', 1.0)
            subsets[var_name] = (data_subset * scale, lats_subset, lons_subset)

        if 'u_wind' in subsets and 'v_wind' in subsets:
            u_data, lats_s, lons_s = subsets['u_wind']
            v_data, _, _ = subsets['v_wind']
            u_earth, v_earth = self._rotate_wind_to_earth(u_data, v_data, lats_s, lons_s)
            subsets['u_wind'] = (u_earth, lats_s, lons_s)
            subsets['v_wind'] = (v_earth, lats_s, lons_s)

        extracted = {}
        for var_name in self.variables.keys():
            data, lats_s, lons_s = subsets[var_name]
            extracted[var_name] = self.interpolate_to_latlon(data, lats_s, lons_s)

        return extracted

    def _validate_variables(self):
        if self.verbose:
            print(f"Validating {len(self.variables)} variables against HRRR...")

        if self.forecast:
            test_date = pd.Timestamp.now(tz='UTC').floor('h').tz_localize(None) - pd.Timedelta(hours=6)
        else:
            test_date = pd.to_datetime(self.start_date.replace('-', ' '))

        try:
            H = Herbie(date=test_date, fxx=0, model='hrrr', product='sfc', verbose=False)
            result = H.xarray(search=self.combined_search, remove_grib=False)

            if isinstance(result, list):
                ds = result[0]
                for other_ds in result[1:]:
                    ds = ds.merge(other_ds, compat='override')
            else:
                ds = result

            available_vars = list(ds.data_vars)

            for var_name, var_info in self.variables.items():
                resolved = self._resolve_var(ds, var_info)
                if resolved:
                    if self.verbose:
                        print(f"  {var_name}: matched '{resolved}'")
                else:
                    ds.close()
                    raise ValueError(
                        f"Variable '{var_name}' (search: {var_info['search']}) "
                        f"did not resolve to any of {var_info['var_names']} "
                        f"or grib_name='{var_info.get('grib_name')}'. "
                        f"Available in dataset: {available_vars}"
                    )

            ds.close()

            if self.verbose:
                print(f"All {len(self.variables)} variables validated successfully.")

        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to validate variables against HRRR: {e}")

    def run_observed_mode(self):
        self.chunk_dir = os.path.join(self.output_dir, 'chunks')
        os.makedirs(self.chunk_dir, exist_ok=True)

        self.final_output = os.path.join(self.output_dir, f'hrrr_surface_{self.grid_size}x{self.grid_size}.npz')

        if os.path.exists(self.final_output) and not self.force_reprocess:
            if self.verbose:
                print(f"Loading existing data from {self.final_output}")
            self.load_final_data()
            return

        if self.verbose:
            print(f"Extracting HRRR surface data (observed)")
            print(f"Period: {self.start_date} to {self.end_date}")
            print(f"Extent: {self.extent}")
            print(f"Output grid: {self.grid_size}x{self.grid_size}")
            print(f"Processing in {self.chunk_months}-month chunks")

        self.process_all_chunks()
        self.combine_chunks()

    def run_historical_forecast_mode(self):
        """Ingest historical HRRR forecast data over a date range.

        For each target hour in [start_date, end_date), finds the most recent
        special run (00/06/12/18 UTC) that is at least 24h before the target,
        then pulls the corresponding forecast hour (fxx=24-29).

        Uses the same chunking, interpolation, and gap-filling as observed mode.
        """
        self.chunk_dir = os.path.join(self.output_dir, 'chunks')
        os.makedirs(self.chunk_dir, exist_ok=True)

        self.final_output = os.path.join(
            self.output_dir, f'hrrr_forecast_{self.grid_size}x{self.grid_size}.npz'
        )

        if os.path.exists(self.final_output) and not self.force_reprocess:
            if self.verbose:
                print(f"Loading existing data from {self.final_output}")
            self.load_final_data()
            return

        if self.verbose:
            print(f"Extracting HRRR historical forecast data (~24h ahead)")
            print(f"Period: {self.start_date} to {self.end_date}")
            print(f"Extent: {self.extent}")
            print(f"Output grid: {self.grid_size}x{self.grid_size}")
            print(f"Processing in {self.chunk_months}-month chunks")

        self.process_all_chunks_historical_forecast()
        self.combine_chunks()

    @staticmethod
    def _target_to_run_and_fxx(target_hour):
        """For a target valid time, find the special run >=24h before and its fxx.

        Returns (run_time, fxx) where run_time is the most recent 00/06/12/18 UTC
        that is at least 24h before target_hour, and fxx is the lead time in hours.
        fxx will range from 24 to 29.
        """
        cutoff = target_hour - pd.Timedelta(hours=24)
        candidate = cutoff.replace(minute=0, second=0, microsecond=0)
        hour = candidate.hour
        special = [0, 6, 12, 18]
        run_hour = max(h for h in special if h <= hour)
        run_time = candidate.replace(hour=run_hour)
        fxx = int((target_hour - run_time).total_seconds() / 3600)
        return run_time, fxx

    def process_all_chunks_historical_forecast(self):
        chunks = self.generate_chunk_periods()
        if self.verbose:
            print(f"\nTotal chunks to process: {len(chunks)}")

        for chunk in tqdm(chunks, desc="Overall progress", disable=not self.verbose):
            chunk_file = os.path.join(self.chunk_dir, chunk['filename'])

            if os.path.exists(chunk_file) and not self.force_reprocess:
                continue

            if self.verbose:
                tqdm.write(f"\n{'='*60}")
                tqdm.write(f"Processing chunk {chunk['num']}/{len(chunks)}")
                tqdm.write(f"Period: {chunk['start']} to {chunk['end']}")

            try:
                chunk_data, timestamps = self.process_chunk_historical_forecast(
                    chunk['start'], chunk['end']
                )

                save_dict = {
                    'timestamps': timestamps,
                    'start_date': chunk['start'],
                    'end_date': chunk['end'],
                }
                save_dict.update(chunk_data)

                np.savez_compressed(chunk_file, **save_dict)
                if self.verbose:
                    tqdm.write(f"Saved chunk {chunk['num']}: {len(timestamps)} timestamps")

                del chunk_data
                gc.collect()

            except Exception as e:
                if self.verbose:
                    tqdm.write(f"Error processing chunk {chunk['num']}: {e}")
                    import traceback
                    traceback.print_exc()
                continue

    def process_chunk_historical_forecast(self, start_date, end_date):
        end_dt = pd.to_datetime(end_date.replace('-', ' '))
        expected_dates = pd.date_range(
            start_date.replace('-', ' '), end_dt, freq='1h', inclusive='left'
        )

        if len(expected_dates) == 0:
            raise ValueError(f"No dates in range {start_date} to {end_date}")

        tasks = []
        for target in expected_dates:
            run_time, fxx = self._target_to_run_and_fxx(target)
            tasks.append((run_time, fxx, target))

        if self.verbose:
            fxx_values = [t[1] for t in tasks]
            tqdm.write(
                f"Forecast lead times: fxx={min(fxx_values)}-{max(fxx_values)}h "
                f"({len(tasks)} target hours)"
            )

        all_data = {var: [] for var in self.variables.keys()}
        successful_dates = []

        n_threads = min(len(tasks), self.max_threads)
        max_attempts = 5

        def download_and_extract(run_time, fxx, target):
            for attempt in range(1, max_attempts + 1):
                try:
                    H = Herbie(
                        date=run_time, fxx=fxx, model='hrrr',
                        product='sfc', verbose=False
                    )
                    if H.grib is None:
                        return target, None
                    H.download(search=self.combined_search)
                    result = H.xarray(search=self.combined_search, remove_grib=False)

                    if isinstance(result, list):
                        ds = result[0]
                        for other_ds in result[1:]:
                            ds = ds.merge(other_ds, compat='override')
                    else:
                        ds = result

                    extracted = self._extract_and_interpolate(ds)
                    ds.close()
                    return target, extracted
                except ConnectionError:
                    if attempt < max_attempts:
                        time.sleep(2 ** attempt)
                        continue
                    return target, None
                except Exception:
                    return target, None
            return target, None

        with ThreadPoolExecutor(n_threads) as exe:
            futures = {
                exe.submit(download_and_extract, rt, fxx, tgt): tgt
                for rt, fxx, tgt in tasks
            }

            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Downloading forecasts", leave=False, disable=not self.verbose
            ):
                target, extracted = future.result()
                if extracted is not None:
                    for var_name in self.variables.keys():
                        all_data[var_name].append(extracted[var_name])
                    successful_dates.append(target)

        sorted_indices = sorted(range(len(successful_dates)), key=lambda i: successful_dates[i])
        successful_dates = [successful_dates[i] for i in sorted_indices]
        for var_name in self.variables.keys():
            all_data[var_name] = np.array([all_data[var_name][i] for i in sorted_indices])

        if self.verbose:
            tqdm.write(f"Downloaded {len(successful_dates)}/{len(expected_dates)} timestamps")

        aligned_data = {}
        for var_name in self.variables.keys():
            aligned_data[var_name] = self.align_and_fill(
                all_data[var_name], successful_dates, expected_dates
            )

        return aligned_data, expected_dates

    def run_forecast_mode(self):
        if self.verbose:
            print(f"Extracting HRRR forecast data")
            print(f"Extent: {self.extent}")
            print(f"Output grid: {self.grid_size}x{self.grid_size}")

        latest_run = self.get_latest_special_run()
        if latest_run is None:
            raise ValueError("No HRRR special run found")

        self.latest_run = latest_run

        if self.verbose:
            print(f"Using HRRR run: {latest_run}")

        now_utc = pd.Timestamp.now(tz='UTC').floor('h').tz_localize(None)
        offset = int((now_utc - latest_run).total_seconds() / 3600)
        fxx_start = offset + 1
        fxx_end = offset + 24

        if self.verbose:
            print(f"Offset: {offset}h, pulling fxx={fxx_start} to fxx={fxx_end}")

        self.download_forecasts(latest_run, fxx_start, fxx_end)

    def get_latest_special_run(self):
        """Find the latest HRRR special run (00/06/12/18 UTC) with full forecast availability."""
        now_utc = pd.Timestamp.now(tz='UTC').floor('h').tz_localize(None)
        special_hours = [18, 12, 6, 0]

        for days_back in [0, 1]:
            date = now_utc - pd.Timedelta(days=days_back)
            for hour in special_hours:
                run_time = date.replace(hour=hour, minute=0, second=0)
                if run_time >= now_utc:
                    continue

                offset = int((now_utc - run_time).total_seconds() / 3600)
                fxx_end = offset + 24

                if fxx_end > 48:
                    if self.verbose:
                        print(f"   Skipping {run_time}: fxx_end={fxx_end} exceeds 48h")
                    continue

                try:
                    H = Herbie(date=run_time, fxx=fxx_end, model='hrrr', product='sfc', verbose=False)
                    if H.grib is not None:
                        if self.verbose:
                            print(f"   Using run {run_time}: fxx={offset+1} to fxx={fxx_end}")
                        return run_time
                except:
                    continue

        return None

    def download_forecasts(self, latest_run, fxx_start, fxx_end):
        fxx_range = list(range(fxx_start, fxx_end + 1))
        forecast_data = {var: [None] * 24 for var in self.variables.keys()}

        def download_forecast_hour(fxx):
            try:
                H = Herbie(date=latest_run, fxx=fxx, model='hrrr', product='sfc', verbose=False)
                H.download(search=self.combined_search)
                result = H.xarray(search=self.combined_search, remove_grib=False)

                if isinstance(result, list):
                    ds = result[0]
                    for other_ds in result[1:]:
                        ds = ds.merge(other_ds, compat='override')
                else:
                    ds = result

                extracted = self._extract_and_interpolate(ds)
                ds.close()
                return fxx, extracted
            except Exception as e:
                if self.verbose:
                    tqdm.write(f"Failed fxx={fxx}: {e}")
                return fxx, None

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {executor.submit(download_forecast_hour, fxx): fxx for fxx in fxx_range}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading forecasts", leave=False):
                fxx, extracted = future.result()
                idx = fxx - fxx_start

                if extracted is not None:
                    for var_name in self.variables.keys():
                        forecast_data[var_name][idx] = extracted[var_name]
                else:
                    for var_name in self.variables.keys():
                        forecast_data[var_name][idx] = np.zeros((self.grid_size, self.grid_size))

        for var_name in self.variables.keys():
            for i in range(24):
                if forecast_data[var_name][i] is None:
                    if i > 0 and forecast_data[var_name][i-1] is not None:
                        forecast_data[var_name][i] = forecast_data[var_name][i-1]
                    else:
                        forecast_data[var_name][i] = np.zeros((self.grid_size, self.grid_size))

        self.data = {}
        for var_name in self.variables.keys():
            self.data[var_name] = np.array(forecast_data[var_name])

        if self.verbose:
            print(f"Complete!")
            for var, arr in self.data.items():
                print(f"  {var}: {arr.shape}, [{arr.min():.2f}, {arr.max():.2f}]")

    def generate_chunk_periods(self):
        chunks = []
        start_dt = pd.to_datetime(self.start_date.replace('-', ' '))
        end_dt = pd.to_datetime(self.end_date.replace('-', ' '))

        current_start = start_dt
        chunk_num = 0

        while current_start < end_dt:
            chunk_num += 1
            current_end = min(current_start + pd.DateOffset(months=self.chunk_months), end_dt)

            chunks.append({
                'num': chunk_num,
                'start': current_start.strftime('%Y-%m-%d-%H'),
                'end': current_end.strftime('%Y-%m-%d-%H'),
                'filename': f"hrrr_chunk_{chunk_num:03d}_{current_start.strftime('%Y%m')}.npz"
            })

            current_start = current_end

        return chunks

    def process_all_chunks(self):
        chunks = self.generate_chunk_periods()
        if self.verbose:
            print(f"\nTotal chunks to process: {len(chunks)}")

        for chunk in tqdm(chunks, desc="Overall progress", disable=not self.verbose):
            chunk_file = os.path.join(self.chunk_dir, chunk['filename'])

            if os.path.exists(chunk_file) and not self.force_reprocess:
                continue

            if self.verbose:
                tqdm.write(f"\n{'='*60}")
                tqdm.write(f"Processing chunk {chunk['num']}/{len(chunks)}")
                tqdm.write(f"Period: {chunk['start']} to {chunk['end']}")

            try:
                chunk_data, timestamps = self.process_chunk(chunk['start'], chunk['end'])

                save_dict = {'timestamps': timestamps, 'start_date': chunk['start'], 'end_date': chunk['end']}
                save_dict.update(chunk_data)

                np.savez_compressed(chunk_file, **save_dict)
                if self.verbose:
                    tqdm.write(f"Saved chunk {chunk['num']}: {len(timestamps)} timestamps")

                del chunk_data
                gc.collect()

            except Exception as e:
                if self.verbose:
                    tqdm.write(f"Error processing chunk {chunk['num']}: {e}")
                    import traceback
                    traceback.print_exc()
                continue

    def process_chunk(self, start_date, end_date):
        end_dt = pd.to_datetime(end_date.replace('-', ' '))
        expected_dates = pd.date_range(start_date.replace('-', ' '), end_dt, freq='1h', inclusive='left')

        if len(expected_dates) == 0:
            raise ValueError(f"No dates in range {start_date} to {end_date}")

        herbies = self.download_all_variables(expected_dates)
        if self.verbose:
            tqdm.write(f"Downloaded {len(herbies)}/{len(expected_dates)} timestamps")

        all_data, successful_dates = self.extract_all_variables(herbies)

        aligned_data = {}
        for var_name in self.variables.keys():
            aligned_data[var_name] = self.align_and_fill(all_data[var_name], successful_dates, expected_dates)

        return aligned_data, expected_dates

    def download_all_variables(self, dates):
        n_threads = min(len(dates), self.max_threads)
        max_attempts = 5

        successful = []
        failed = []

        with ThreadPoolExecutor(n_threads) as exe:
            def download_task(date):
                for attempt in range(1, max_attempts + 1):
                    try:
                        H = Herbie(date=date, fxx=0, verbose=False)
                        if H.grib is not None:
                            H.download(search=self.combined_search)
                            return H
                        return None
                    except ConnectionError:
                        if attempt < max_attempts:
                            time.sleep(2 ** attempt)
                            continue
                        return None
                    except Exception:
                        return None
                return None

            futures = {exe.submit(download_task, date): date for date in dates}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading", leave=False, disable=not self.verbose):
                result = future.result()
                if result is not None:
                    successful.append(result)
                else:
                    failed.append(futures[future])

        successful.sort(key=lambda H: H.date)

        if failed and self.verbose:
            tqdm.write(f"Failed to download {len(failed)} timestamps")

        return successful

    def extract_all_variables(self, herbie_list):
        all_data = {var: [] for var in self.variables.keys()}
        successful_dates = []

        for H in tqdm(herbie_list, desc="Extracting", leave=False, disable=not self.verbose):
            try:
                result = H.xarray(search=self.combined_search, remove_grib=False)

                if isinstance(result, list):
                    ds = result[0]
                    for other_ds in result[1:]:
                        ds = ds.merge(other_ds, compat='override')
                else:
                    ds = result

                extracted = self._extract_and_interpolate(ds)
                ds.close()

                for var_name in self.variables.keys():
                    all_data[var_name].append(extracted[var_name])
                successful_dates.append(H.date)

            except Exception as e:
                if self.verbose:
                    tqdm.write(f"Failed to extract {H.date}: {e}")
                continue

        for var_name in all_data.keys():
            if len(all_data[var_name]) > 0:
                all_data[var_name] = np.array(all_data[var_name])
            else:
                all_data[var_name] = np.array([]).reshape(0, self.grid_size, self.grid_size)

        return all_data, successful_dates

    def subset_and_get_coords(self, ds, var_name):
        data = ds[var_name].values

        if data.ndim == 3:
            data = data[0]

        if 'latitude' in ds.coords:
            lats = ds['latitude'].values
            lons = ds['longitude'].values
        else:
            lats = ds['lat'].values
            lons = ds['lon'].values

        if lats.ndim == 2:
            lon_min = 360 + self.extent[0]
            lon_max = 360 + self.extent[1]

            lat_mask = (lats >= self.extent[2]) & (lats <= self.extent[3])
            lon_mask = (lons >= lon_min) & (lons <= lon_max)
            combined_mask = lat_mask & lon_mask

            where_result = np.where(combined_mask)
            if len(where_result[0]) > 0:
                i_min, i_max = where_result[0].min(), where_result[0].max()
                j_min, j_max = where_result[1].min(), where_result[1].max()

                pad = 2
                i_min = max(0, i_min - pad)
                i_max = min(data.shape[0] - 1, i_max + pad)
                j_min = max(0, j_min - pad)
                j_max = min(data.shape[1] - 1, j_max + pad)

                data_subset = data[i_min:i_max+1, j_min:j_max+1]
                lats_subset = lats[i_min:i_max+1, j_min:j_max+1]
                lons_subset = lons[i_min:i_max+1, j_min:j_max+1] - 360

                return data_subset, lats_subset, lons_subset

        return data, lats, lons - 360

    def interpolate_to_latlon(self, data, lats, lons):
        points = np.column_stack((lons.flatten(), lats.flatten()))
        values = data.flatten()

        valid_mask = ~np.isnan(values)
        if not np.any(valid_mask):
            return np.zeros((self.grid_size, self.grid_size))

        points = points[valid_mask]
        values = values[valid_mask]

        try:
            interpolated = griddata(points, values, (self.lon_grid, self.lat_grid), method='linear')

            if np.any(np.isnan(interpolated)):
                interpolated_nn = griddata(points, values, (self.lon_grid, self.lat_grid), method='nearest')
                interpolated = np.where(np.isnan(interpolated), interpolated_nn, interpolated)
        except:
            interpolated = griddata(points, values, (self.lon_grid, self.lat_grid), method='nearest')

        return np.flip(interpolated, axis=0)

    def align_and_fill(self, frames, dates, expected_dates):
        n_times = len(expected_dates)
        aligned = np.full((n_times, self.grid_size, self.grid_size), np.nan)

        for frame, date in zip(frames, dates):
            try:
                idx = expected_dates.get_loc(date)
                aligned[idx] = frame
            except KeyError:
                continue

        missing = np.isnan(aligned).all(axis=(1, 2)).sum()
        if missing > 0 and self.verbose:
            tqdm.write(f"Data gaps: {missing} hours missing, forward filling...")

        for t in range(n_times):
            if np.isnan(aligned[t]).all() and t > 0:
                aligned[t] = aligned[t-1]

        if np.isnan(aligned[0]).all():
            for t in range(1, n_times):
                if not np.isnan(aligned[t]).all():
                    aligned[0] = aligned[t]
                    break

        return aligned

    def combine_chunks(self):
        if self.verbose:
            print(f"\nCombining all chunks into final output...")

        chunks = self.generate_chunk_periods()

        all_vars = list(self.variables.keys())
        all_data = {var: [] for var in all_vars}
        all_timestamps = []

        for chunk in tqdm(chunks, desc="Combining chunks", disable=not self.verbose):
            chunk_file = os.path.join(self.chunk_dir, chunk['filename'])

            if not os.path.exists(chunk_file):
                if self.verbose:
                    tqdm.write(f"Warning: Missing chunk {chunk['num']}, skipping...")
                continue

            data = np.load(chunk_file, allow_pickle=True)

            for var in all_vars:
                if var in data:
                    all_data[var].append(data[var])

            all_timestamps.extend(data['timestamps'])

        final_data = {}
        for var, arrays in all_data.items():
            if len(arrays) > 0:
                final_data[var] = np.concatenate(arrays, axis=0)

        if self.verbose:
            print(f"Saving final dataset to {self.final_output}")

        save_dict = {
            'timestamps': all_timestamps,
            'extent': self.extent,
            'grid_size': self.grid_size,
            'variables': list(final_data.keys())
        }
        save_dict.update(final_data)

        np.savez_compressed(self.final_output, **save_dict)

        if self.verbose:
            print(f"\nComplete!")
            print(f"Timestamps: {len(all_timestamps)} | Range: {all_timestamps[0]} to {all_timestamps[-1]}")
            for var, arr in final_data.items():
                print(f"  {var}: {arr.shape}, [{arr.min():.2f}, {arr.max():.2f}]")

        self.data = final_data
        self.timestamps = all_timestamps

    def load_final_data(self):
        data = np.load(self.final_output, allow_pickle=True)

        self.data = {}
        for key in data.files:
            if key not in ['timestamps', 'extent', 'grid_size', 'variables']:
                self.data[key] = data[key]

        self.timestamps = data['timestamps']

        if self.verbose:
            print(f"Loaded: {len(self.timestamps)} timestamps")
            for var, arr in self.data.items():
                print(f"  {var}: {arr.shape}, [{arr.min():.2f}, {arr.max():.2f}]")


if __name__ == "__main__":
    extent = (-118.615, -117.70, 33.60, 34.35)

    print("Testing observed mode...")
    extractor = HRRRData(
        start_date="2023-08-02-00",
        end_date="2023-08-03-00",
        extent=extent,
        extent_name='la_basin',
        output_dir='data/hrrr_test',
        grid_size=84,
        chunk_months=2,
        force_reprocess=True,
        verbose=True,
        max_threads=20,
        forecast=False,
    )

    print("\nTesting forecast mode...")
    forecast_extractor = HRRRData(
        extent=extent,
        output_dir='data/hrrr_forecast_test',
        grid_size=84,
        force_reprocess=True,
        verbose=True,
        max_threads=10,
        forecast=True,
    )