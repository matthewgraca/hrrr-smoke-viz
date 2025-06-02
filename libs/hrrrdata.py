from herbie import Herbie, wgrib2
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np
import xarray as xr
import cv2
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import ConnectionError 
import subprocess
from shutil import which
import os
import gc

class HRRRData:
    def __init__(
        self,
        start_date,
        end_date,
        extent=None, 
        extent_name='subset_region',
        product='MASSDEN',
        frames_per_sample=1,
        dim=40,
        sample_setting=1,
        verbose=False,
        processed_cache_dir='data/hrrr_processed.npz',
        chunk_cache_dir='data/hrrr_chunks',
        force_reprocess=False,
        chunk_months=1
    ):
        '''
        Gets the HRRR data.
        Pipeline:
            - Use Herbie to download the data as grib files
            - Use wgrib2 to subregion the grib files
            - Convert the grib files to xarray, then into numpy as frames
                - (num_frames, row, col)
            - Interpolate and add a channel axis to the array of frames
                - (num_frames, row, col, channel)
            - Create samples from a sliding window of frames
                - (samples, frames, row, col, channel)

        Members:
            data: The complete processed HRRR data
        '''
        self.chunk_months = chunk_months
        self.chunk_cache_dir = chunk_cache_dir
        
        # Create chunk cache directory
        if chunk_cache_dir is not None:
            os.makedirs(chunk_cache_dir, exist_ok=True)
        
        # read from final cache if enabled
        if processed_cache_dir is not None:
            os.makedirs(os.path.dirname(processed_cache_dir), exist_ok=True)

            cache_exists = os.path.exists(processed_cache_dir)
            if not force_reprocess and cache_exists: 
                print(
                    f"📖 Loading processed HRRR data from final cache: "
                    f"{processed_cache_dir}"
                )
                try:
                    # attempt to read from cache
                    cached_data = np.load(
                        processed_cache_dir, 
                        allow_pickle=True
                    )
                    self.data = cached_data['data']
                    cached_start, cached_end = cached_data['date_range']
                    cached_product = cached_data['product']
                    cached_extent = cached_data['extent']
                    cached_sample_setting = cached_data['sample_setting']

                    print(f"🎉 Successfully loaded data from final cache:")
                    print(
                        f"  - Data shape    : {self.data.shape}\n"
                        f"  - Date range    : [{cached_start}, "
                        f"{cached_end})\n"
                        f"  - Product used  : {cached_product}\n"
                        f"  - Extent        : {cached_extent}\n"
                        f"  - Sample setting: {cached_sample_setting}"
                    )
                    return # return early if loading cache is successful
                except Exception as e:
                    print(
                        f"❗ Error loading from final cache: {e}. "
                        f"Will check for chunk caches or reprocess data.")
            else:
                print(
                    f"🔎 Either final cache is empty, or force_reprocess flag "
                    f"was raised. Checking for chunk caches..."
                )

        print(f"🗓️ Processing HRRR data in {chunk_months}-month chunks from {start_date} to {end_date}")
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        chunk_info = self._generate_chunk_list(start_dt, end_dt, chunk_months)
        
        existing_chunks, missing_chunks = self._check_chunk_cache_status(chunk_info, chunk_cache_dir, force_reprocess)
        
        if existing_chunks:
            print(f"📦 Found {len(existing_chunks)} existing cached chunks")
        if missing_chunks:
            print(f"⚙️ Need to process {len(missing_chunks)} missing chunks")
            
            # Process missing chunks
            self._process_missing_chunks(
                missing_chunks, extent, extent_name, product, 
                frames_per_sample, dim, sample_setting, verbose
            )
        else:
            print("✅ All chunks already cached!")

        print(f"🔗 Loading and combining all {len(chunk_info)} chunks...")
        self.data = self._load_and_combine_chunks(chunk_info)
        print(f"🎉 Final combined data shape: {self.data.shape}")

        if processed_cache_dir is None:
            print("🙅 No final cache directory set. Combined data will not be cached.")
        else:
            print(f"💾 Saving final combined data to cache: {processed_cache_dir}")
            try:
                np.savez_compressed(
                    processed_cache_dir,
                    data=self.data,
                    date_range=np.array([start_date, end_date]),
                    product=np.array([product]),
                    extent=np.array(extent) if extent is not None else np.array([]),
                    sample_setting=np.array([sample_setting])
                )
                print("🎉 Successfully saved final combined data to cache.")
            except Exception as e:
                print(f"⚠️  Warning: Could not save final combined data to cache: {e}")

    def _generate_chunk_list(self, start_dt, end_dt, chunk_months):
        """Generate list of chunk information (start, end, filename)."""
        chunks = []
        current_start = start_dt
        chunk_num = 0
        
        while current_start < end_dt:
            chunk_num += 1
            current_end = min(
                current_start + pd.DateOffset(months=chunk_months), 
                end_dt
            )
            
            chunk_filename = f"chunk_{chunk_num:03d}_{current_start.strftime('%Y%m%d')}_{current_end.strftime('%Y%m%d')}.npz"
            
            chunks.append({
                'num': chunk_num,
                'start': current_start,
                'end': current_end,
                'start_str': current_start.strftime('%Y-%m-%d-%H'),
                'end_str': current_end.strftime('%Y-%m-%d-%H'),
                'filename': chunk_filename
            })
            
            current_start = current_end
            
        return chunks

    def _check_chunk_cache_status(self, chunk_info, chunk_cache_dir, force_reprocess):
        """Check which chunks exist in cache and which need processing."""
        existing_chunks = []
        missing_chunks = []
        
        for chunk in chunk_info:
            chunk_path = os.path.join(chunk_cache_dir, chunk['filename'])
            
            if not force_reprocess and os.path.exists(chunk_path):
                try:
                    test_load = np.load(chunk_path)
                    if 'data' in test_load:
                        existing_chunks.append(chunk)
                        print(f"✅ Chunk {chunk['num']}: {chunk['filename']} (cached)")
                    else:
                        missing_chunks.append(chunk)
                        print(f"❌ Chunk {chunk['num']}: Invalid cache file, will reprocess")
                except Exception as e:
                    missing_chunks.append(chunk)
                    print(f"❌ Chunk {chunk['num']}: Cache corrupt ({e}), will reprocess")
            else:
                missing_chunks.append(chunk)
                status = "forced reprocess" if force_reprocess else "not cached"
                print(f"⏳ Chunk {chunk['num']}: {chunk['filename']} ({status})")
        
        return existing_chunks, missing_chunks

    def _process_missing_chunks(self, missing_chunks, extent, extent_name, 
                              product, frames_per_sample, dim, sample_setting, verbose):
        """Process and cache missing chunks individually."""
        for chunk in missing_chunks:
            print(f"\n📦 Processing chunk {chunk['num']}: {chunk['start_str']} to {chunk['end_str']}")
            
            try:
                chunk_data = self._process_chunk(
                    chunk['start_str'], chunk['end_str'],
                    extent, extent_name, product, frames_per_sample, 
                    dim, sample_setting, verbose
                )
                
                if chunk_data is not None and len(chunk_data) > 0:
                    chunk_path = os.path.join(self.chunk_cache_dir, chunk['filename'])
                    np.savez_compressed(
                        chunk_path,
                        data=chunk_data,
                        start_date=chunk['start_str'],
                        end_date=chunk['end_str'],
                        chunk_num=chunk['num']
                    )
                    print(f"✅ Chunk {chunk['num']} processed and cached: {chunk_data.shape}")
                else:
                    print(f"⚠️ Chunk {chunk['num']} returned no data")
                    
            except Exception as e:
                print(f"❌ Error processing chunk {chunk['num']}: {e}")
                
            gc.collect()

    def _load_and_combine_chunks(self, chunk_info):
        """Load all cached chunks and combine them into final dataset."""
        all_chunks = []
        
        for chunk in chunk_info:
            chunk_path = os.path.join(self.chunk_cache_dir, chunk['filename'])
            
            try:
                print(f"📖 Loading chunk {chunk['num']}: {chunk['filename']}")
                cached_chunk = np.load(chunk_path)
                chunk_data = cached_chunk['data']
                all_chunks.append(chunk_data)
                print(f"   ✅ Loaded chunk {chunk['num']}: {chunk_data.shape}")
                
            except Exception as e:
                print(f"❌ Error loading chunk {chunk['num']}: {e}")
                raise ValueError(f"Failed to load required chunk {chunk['num']}. Try reprocessing with force_reprocess=True")
        
        if not all_chunks:
            raise ValueError("No chunks were successfully loaded")
            
        # Combine all chunks
        print("🔗 Combining all chunks...")
        combined_data = np.concatenate(all_chunks, axis=0)
        
        # Clear chunks from memory
        del all_chunks
        import gc
        gc.collect()
        
        return combined_data

    def _process_chunk(
        self, 
        start_date, 
        end_date, 
        extent, 
        extent_name, 
        product, 
        frames_per_sample, 
        dim, 
        sample_setting, 
        verbose
    ):
        """Process a single chunk of HRRR data."""
        try:
            # frame-by-frame sampling
            if sample_setting == 1:
                herbie_ds = self._get_hrrr_data_frame_by_frame(
                    start_date, end_date, product, verbose
                )
                
                if not herbie_ds:
                    print(f"No herbie data for chunk {start_date} to {end_date}")
                    return None
                    
                subregion_grib_ds = self._subregion_grib_files(
                    herbie_ds, extent, extent_name, product
                )
                
                if not subregion_grib_ds:
                    print(f"No subregion data for chunk {start_date} to {end_date}")
                    return None
                    
                subregion_frames = self._grib_to_np(subregion_grib_ds)
                
                if len(subregion_frames) == 0:
                    print(f"No frames extracted for chunk {start_date} to {end_date}")
                    return None
                    
                preprocessed_frames = self._interpolate_and_add_channel_axis(
                    subregion_frames, dim
                )
                processed_ds = self._sliding_window_of(
                    preprocessed_frames, frames_per_sample
                )

            # offset-by-sample with forecast sampling
            elif sample_setting == 2:
                herbie_ds = self._get_hrrr_data_offset_by_forecast(
                    start_date, end_date, frames_per_sample, product, verbose
                )
                
                if not herbie_ds:
                    print(f"No herbie data for chunk {start_date} to {end_date}")
                    return None
                    
                subregion_grib_ds = self._subregion_grib_files(
                    herbie_ds, extent, extent_name, product
                )
                
                if not subregion_grib_ds:
                    print(f"No subregion data for chunk {start_date} to {end_date}")
                    return None
                    
                subregion_frames = self._grib_to_np(subregion_grib_ds)
                
                if len(subregion_frames) == 0:
                    print(f"No frames extracted for chunk {start_date} to {end_date}")
                    return None
                    
                preprocessed_frames = self._interpolate_and_add_channel_axis(
                    subregion_frames, dim
                )
                # sliding window needs to be offset by the number of frames
                processed_ds = self._sliding_window_of(
                    frames=preprocessed_frames, 
                    window_size=frames_per_sample, 
                    full_slide=True 
                )
            else:
                msg = (
                    "Argument \"sample_setting\" must be either:\n",
                    "1 - frame-by-frame\n",
                    "2 - offset-by-sample with forecasts\n"
                )
                raise ValueError(" ".join(msg))

            return processed_ds
            
        except Exception as e:
            print(f"Error in _process_chunk: {e}")
            return None

    def _attempt_download(
        self, 
        date_range, 
        product, 
        forecast_range=[0], 
        max_threads=20
    ):
        '''
        FastHerbie, but it throws exceptions instead of just logging them.
        call it furbie, lmk
        - Will handle reset connection (104), goes for 5 max attempts
        - Any unhandled exception will throw it up.

        Returns a list of Herbie objects.
        '''

        n_tasks = len(date_range) * len(forecast_range)
        n_threads = min(n_tasks, max_threads)

        # multithread locating grib files
        # will pump out a list of herbie objects
        herbies = self._multithread_dl_herbie_objects(
            date_range, 
            forecast_range, 
            n_threads
        )

        # notify user about grib files that can't be found;
        # not really planning on doing anything about though
        found_files = [H for H in herbies if H.grib is not None]
        lost_files = [H for H in herbies if H.grib is None]
        if len(lost_files) > 0:
            print(
                f"⚠️  Could not find the following grib files:\n"
                f"{lost_files}"
            )

        # multithread download grib files 
        outfiles = self._multithread_dl_grib_files(
            found_files, 
            product, 
            n_threads
        )

        # confirm xarray can read the grib files
        for H in found_files:
            self._verify_downloads(H, product)

        return found_files

    def _verify_downloads(self, H, product):
        """
        Will attempt to open the grib file with xarray.
        If it cannot be opened, it will trigger a redownload

        If successful, will continue. If unsuccessful, will 
        attempt to redownload 5 times until re-raising the error.
        """
        success = False
        attempt = 1
        max_attempts = 5
        while not success:
            try:
                file = H.get_localFilePath(product)
                xr.open_dataset(file, engine="cfgrib", decode_timedelta=False)
                success = True
            except Exception as e:
                # primarily meant to catch EOFError, but all errors should
                if attempt > max_attempts:
                    print("🚨 Max attempts reached, raising error.")
                    raise

                print(
                    f"{e}\n"
                    f"🤕 File corrupted while downloading; "
                    f"beginning redownload attempt "
                    f"{attempt}/{max_attempts}."
                )
                self._redownload(H, product, file)
                attempt += 1

    # NOTE downloading helpers
    def _multithread_dl_grib_files(self, herbie_data, product, n_threads):
        with ThreadPoolExecutor(n_threads) as exe:
            def dl_grib_task(H, product):
                return self._download_thread(H.download, search=product)
            futures = [
                exe.submit(dl_grib_task, H, product) 
                for H in herbie_data 
            ]
        outfiles = [future.result() for future in as_completed(futures)]

        return outfiles
    
    def _multithread_dl_herbie_objects(
        self, 
        date_range, 
        forecast_range, 
        n_threads
    ):
        '''
        Finds the remote grib files using Herbie. The Herbie object contains
        a bunch of metadata regarding the grib file.

        Also, the herbie objects will be sorted by date and forecast time.
        '''
        with ThreadPoolExecutor(n_threads) as exe:
            def herbie_task(date_step, forecast_step):
                return self._download_thread(
                    Herbie, date=date_step, fxx=forecast_step
                )
            futures = [
                exe.submit(herbie_task, date_step, forecast_step)
                for date_step in date_range
                for forecast_step in forecast_range
            ]
        herbies = [future.result() for future in as_completed(futures)]
        herbies.sort(key=lambda H: H.fxx)
        herbies.sort(key=lambda H: H.date)

        return herbies

    def _download_thread(self, dl_task, **kwargs):
        '''
        The function responsible for performing the download.

        If there are no exceptions, the task to download will be run 
        Will catch 104: Connection reset, and retry up to 5 times.
        '''
        max_attempts = 5
        delay = 2
        for attempt in range(2, max_attempts+2):
            try:
                return dl_task(**kwargs)
            except ConnectionError as e:
                if attempt <= max_attempts:
                    print(
                        f"⚠️  Error while downloading: {e}\n"
                        f"🔧 Attempt number {attempt}/{max_attempts}, "
                        f"backing off by {delay} seconds."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    print(
                        "❌ Download failed after {max_attempts} attempts!"
                    )
                    raise
            except Exception as e:
                print(f"Unknown exception: {e}")
                raise
        # shouldn't be here. either throws error or returns the task.

    def _get_hrrr_data_frame_by_frame(
        self, 
        start_date, 
        end_date, 
        product, 
        verbose
    ):
        '''
        Uses FastHerbie to grab the remote data, and download it; frame by frame.

        Arguments:
            start_date: The start date of the query, in the form "yyyy-mm-dd-hh"
            end_date: The end date of the query (exclusive)
            product: regex of the product to download 
            verbose: Determines if Herbie objects should be printed

        Returns:
            The list of Herbie objects of the downloaded data
        '''
        end_date = pd.to_datetime(end_date) - pd.Timedelta(hours=1)
        dates = pd.date_range(start_date, end_date, freq="1h")

        if len(dates) == 0:
            print(f"No dates in range {start_date} to {end_date}")
            return []

        FH = self._attempt_download(
            date_range=dates, 
            product=product, 
            forecast_range=[0]
        )

        if verbose:
            [print(repr(H)) for H in FH]

        return FH

    def _get_hrrr_data_offset_by_forecast(
        self, 
        start_date,
        end_date,
        offset,
        product,
        verbose
    ):
        '''
        Uses FastHerbie to grab the remote data, and download it; offset by the
        number of frames per sample, and using forecasts.

        Arguments:
            start_date: The start date of the query, in the form "yyyy-mm-dd-hh"
            end_date: The end date of the query (exclusive)
            offset: The number of frames per sample we offset by
            product: regex of the product to download 
            verbose: Determines if Herbie objects should be printed

        Returns:
            The list of Herbie objects of the downloaded data

        '''
        # if sample is t=0, pull init @ 00, fxx = 01 so we provide next-sample forecast
        offset_start_date = pd.to_datetime(start_date) + pd.Timedelta(hours=offset - 1)
        end_date = pd.to_datetime(end_date) - pd.Timedelta(hours=1)
        dates = pd.date_range(offset_start_date, end_date, freq="1h")

        if len(dates) == 0:
            print(f"No dates in range {offset_start_date} to {end_date}")
            return []

        FH = self._attempt_download(
            date_range=dates,
            product=product,
            forecast_range=[i for i in range(1, offset + 1)]
        )

        if verbose:
            [print(repr(H)) for H in FH]

        return FH

    def _subregion_grib_files(self, herbie_data, extent, extent_name, product):
        '''
        Takes a list of Herbie objects, and subregions the downloaded grib 
        files. If there is no defined extent, no changes will be made.

        Arguments:
            herbie_data: The list containing the Herbie objects
            extent: The bounding box with the shape: (a, b, c, d):
                - a: bottom longitude
                - b: top longitude
                - c: bottom latitude
                - d: top latitude
            extent_name: Desired name of the subregion
            product: Regex of the product that was downloaded

        Returns:
            A list of the subregioned grib files
        '''
        subregion_grib_files = []
        for H in herbie_data:
            attempts = 1
            max_attempts = 3
            success = False
            while not success:
                try:
                    file = H.get_localFilePath(product)
                    idx_file = wgrib2.create_inventory_file(file)
                    subset_file = (
                        file if extent is None
                        else wgrib2.region(file, extent, name=extent_name)
                    )
                    subregion_grib_files.append(subset_file)
                    success = True
                except subprocess.CalledProcessError as e:
                    if attempts > max_attempts:
                        print("🚨 Max attempts reached, raising error.")
                        raise

                    print(
                        f"⚠️  Issue found with file {file}.\n"
                        f"wgrib2 exit code: {e.returncode}, "
                        f"with error: {e.stderr}\n"
                        f"Attempt {attempts}/{max_attempts}: "
                        f"redownload and running subregion."
                    )

                    self._redownload(H, product, file)
                    attempts += 1
                except Exception as e:
                    print(f"Unknown exception: {e}")
                    raise

        return subregion_grib_files

    def _redownload(self, H, product, file):
        '''
        Helper to subregion(), which will delete the an offending file
        and use the Herbie object + the product to redownload the file
        '''
        p = subprocess.run(
            f"{which('rm')} {file}",
            shell=True,
            capture_output=True,
            encoding="utf-8",
            check=True,
        )

        # if file DNE, notify user (but do nothing about it)
        if p.returncode != 0:
            print(f"Warning: {p.stderr}")

        # the issue here is the new file may be different;
        # idk if redownloading changes the name? it seems that for
        # the exact same subset, the file doesn't change; that's a relief..
        self._multithread_dl_grib_files(
            herbie_data=[H], 
            product=product, 
            n_threads=1
        )

    def _grib_to_np(self, grib_files):
        '''
        Converts a list of downloaded grib files into numpy. We assume that
        there is only one data variable in the grib file.

        Arguments:
            grib_files: The paths of the grib files being converted

        Returns:
            A numpy array of the grib files
        '''
        np_of_grib = []
        for file in grib_files:
            try:
                hrrr_xarr = xr.open_dataset(file, engine="cfgrib", decode_timedelta=False)

                # for now, we'll only accept Datasets with one variable
                data_vars = [variable for variable in hrrr_xarr.data_vars]
                if len(data_vars) != 1:
                    err_msg = (
                        f"xarray Dataset should have only one data variable."
                        f"Expected: 'unknown' for 'COLMD' or 'mdens' for 'MASSDEN',"
                        f"Returned: {hrrr_xarr.data_vars.keys()}"
                    )
                    print(f"Warning: {' '.join(err_msg)} for file {file}")
                    continue
                variable = data_vars[0]

                # hrrr data comes in (y, x), so it will need to be flipped for (x, y)
                np_of_grib.append(np.flip(hrrr_xarr[variable].to_numpy(), axis=0))
                
                # Close the dataset to free memory
                hrrr_xarr.close()
                
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue

        return np.array(np_of_grib)

    def _interpolate_and_add_channel_axis(self, subregion_ds, dim):
        ''' 
        Takes a dataset frames (x, y), interpolates/decimates according to
        the dimensions, and adds a channel axis.

        Arguments:
            subregion_ds: The numpy array containing a list of frames
            dim: The desired dimensions

        Returns:
            A numpy array of frames
        '''
        n_frames = len(subregion_ds)
        channels = 1
        frames = np.empty(shape=(n_frames, dim, dim, channels))

        # interpolate and add channel axis
        for i, frame in enumerate(subregion_ds):
            new_frame = cv2.resize(frame, (dim, dim))
            new_frame = np.reshape(new_frame, (dim, dim, channels))
            frames[i] = new_frame

        return frames

    def _sliding_window_of(self, frames, window_size, full_slide=False):
        '''
        Uses a sliding window to create samples from frames.

        For example, with a step size of 5, with 15 frames and a 5 frames per 
        sample, frames: 
            - 1-5
            - 6-10
            - 11-15
        will make up a total of 3 samples.

        For example, with a step size of 1, with 5 frames and a 3 frames per 
        sample, frames: 
            - 1-3 
            - 2-4
            - 3-5 
        will make up a total of 3 samples.

        Arguments:
            frames: A numpy array of the shape (num_frames, row, col, channels)
            window_size: The desired number of frames per sample
            full_slide: Whether the window should slide by 1 (false) or by the size
                of the window (true)

        Returns:
            A numpy array of the shape (num_samples, num_frames, row, col, channels)
        '''
        n_frames, row, col, channels = frames.shape
        n_samples = (
            n_frames - window_size + 1
            if full_slide == False
            else n_frames // window_size
        )

        if n_samples <= 0:
            print(f"Warning: Not enough frames ({n_frames}) for window size ({window_size})")
            return np.array([])

        samples = np.empty((n_samples, window_size, row, col, channels))

        for i in range(n_samples):
            start_idx = i if full_slide == False else i * window_size 
            end_idx = start_idx + window_size

            samples[i] = np.array(
                [frames[j] for j in range(start_idx, end_idx)]
            )

        return samples