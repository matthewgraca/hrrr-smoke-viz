import numpy as np
import pandas as pd
import cv2
import requests
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tarfile
import urllib.request
from bs4 import BeautifulSoup
import json
import time
import warnings
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

# Try to import GDAL, but continue if not available
try:
    from osgeo import gdal
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    print("Warning: GDAL not available. Required for MAIAC data processing.")

# Try to import pyhdf, but continue if not available
try:
    from pyhdf.SD import SD, SDC
    PYHDF_AVAILABLE = True
except ImportError:
    PYHDF_AVAILABLE = False
    print("Warning: pyhdf not available. May be needed as fallback for MAIAC data.")

# Try to import skimage.io, but continue if not available
try:
    import skimage.io as io
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: skimage.io not available. Will use alternative methods for image reading.")

class PWWBData:
    def __init__(
        self,
        start_date,
        end_date,
        extent=(-118.75, -117.5, 33.5, 34.5),
        frames_per_sample=5,
        dim=200,
        cache_dir='data/pwwb_cache/',
        use_cached_data=True,
        verbose=False,
        env_file='.env',
        include_wind=False  # New parameter to control wind data inclusion
    ):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.extent = extent
        self.frames_per_sample = frames_per_sample
        self.dim = dim
        self.verbose = verbose
        self.cache_dir = cache_dir
        self.use_cached_data = use_cached_data
        self.include_wind = include_wind  # Store the parameter
        
        # Load environment variables
        if env_file and os.path.exists(env_file):
            load_dotenv(env_file)
        
        # Get credentials from environment
        self.m2m_credentials = {
            'username': os.getenv('M2M_USERNAME'),
            'token': os.getenv('M2M_TOKEN')
        }
        
        self.maiac_credentials = {
            'username': os.getenv('MAIAC_USERNAME'),
            'password': os.getenv('MAIAC_PASSWORD')
        }
        
        # Only get OpenWeatherMap API key if include_wind is True
        self.openweather_api_key = os.getenv('OPENWEATHER_API_KEY') if include_wind else None
        
        os.makedirs(cache_dir, exist_ok=True)
        
        self.timestamps = pd.date_range(self.start_date, self.end_date, freq='H')
        self.n_timestamps = len(self.timestamps)
        
        self._process_pipeline()
    
    def _process_pipeline(self):
        channels = {}
        
        if self.verbose:
            print("Processing Landsat data...")
        channels['landsat'] = self._get_landsat_data()
        
        if self.verbose:
            print("Processing MAIAC AOD data...")
        channels['maiac'] = self._get_maiac_data()
        
        # Only include wind data if specifically requested
        has_wind_data = False
        if self.include_wind and self.openweather_api_key:
            if self.verbose:
                print("Processing wind data...")
            try:
                channels['wind'] = self._get_wind_data()
                has_wind_data = True
            except Exception as e:
                if self.verbose:
                    print(f"Wind data processing failed: {e}")
                    print("Continuing without wind data...")
        else:
            if self.verbose:
                print("Wind data processing skipped.")
        
        if self.verbose:
            print("Normalizing channels...")
        normalized_channels = self._normalize_channels(channels)
        
        # Concatenate channels based on what's available
        channel_list = [
            normalized_channels['landsat'],  # 3 channels
            normalized_channels['maiac'],    # 1 channel
        ]
        
        # Add wind data if available
        if has_wind_data:
            channel_list.append(normalized_channels['wind'])  # 1 channel
        
        self.all_channels = np.concatenate(channel_list, axis=-1)
        
        # Visualize combined channels
        if self.verbose:
            channel_names = self.get_channel_info()['channel_order']
            self.visualize_combined_data(
                np.expand_dims(np.expand_dims(self.all_channels, 0), 0),  # Add batch and time dimensions
                channel_names
            )
        
        self.data = self._sliding_window_of(self.all_channels, self.frames_per_sample)
        
        self.landsat_data = channels['landsat']
        self.maiac_data = channels['maiac']
        if has_wind_data:
            self.wind_data = channels['wind']
        else:
            self.wind_data = None
        
        # Track which channels are included
        self.has_wind_data = has_wind_data
        
        if self.verbose:
            print(f"Final data shape: {self.data.shape}")
    
    def _get_landsat_data(self):
        cache_file = os.path.join(self.cache_dir, 'landsat_data.npy')
        
        if self.use_cached_data and os.path.exists(cache_file):
            if self.verbose:
                print(f"Loading cached Landsat data from {cache_file}")
            landsat_data = np.load(cache_file)
            
            # Add debugging info
            if self.verbose:
                print(f"Landsat data shape: {landsat_data.shape}")
                print(f"Landsat data stats: min={landsat_data.min():.3f}, max={landsat_data.max():.3f}, mean={landsat_data.mean():.3f}")
                
                # Check mask percentage (percentage of zeros or very small values)
                mask_percentage = np.mean(landsat_data < 0.1) * 100
                print(f"Landsat mask percentage (values < 0.1): {mask_percentage:.2f}%")
                
                # Print per-channel stats
                for c in range(landsat_data.shape[-1]):
                    channel = landsat_data[0, :, :, c]  # First frame, channel c
                    print(f"  Channel {c} stats: min={channel.min():.3f}, max={channel.max():.3f}, mean={channel.mean():.3f}")
                    
                    # Calculate percentage of the frame that has data
                    data_percentage = np.mean(channel > 0.1) * 100
                    print(f"  Channel {c} data coverage: {data_percentage:.2f}%")
                    
                    # Generate a heatmap of the data
                    plt.figure(figsize=(8, 8))
                    plt.imshow(channel, cmap='viridis')
                    plt.colorbar(label='Value')
                    plt.title(f'Landsat Channel {c} Data')
                    plt.savefig(f'landsat_ch{c}_heatmap.png')
                    plt.close()
                    print(f"  Channel {c} heatmap saved to landsat_ch{c}_heatmap.png")
            
            return landsat_data
        
        if not self.m2m_credentials['username'] or not self.m2m_credentials['token']:
            raise ValueError("M2M credentials not found in environment variables. Set M2M_USERNAME and M2M_TOKEN.")
        
        products_available = self._fetch_landsat_from_usgs()
        if not products_available:
            raise RuntimeError("No Landsat products available for the specified date range and extent.")
        
        landsat_raw = self._process_landsat_products(products_available)
        
        # Add debugging info for raw Landsat data
        if self.verbose:
            print(f"Raw Landsat data shape: {landsat_raw.shape}")
            print(f"Raw Landsat data stats: min={landsat_raw.min():.3f}, max={landsat_raw.max():.3f}, mean={landsat_raw.mean():.3f}")
            
            # Generate heatmaps for raw data
            for c in range(landsat_raw.shape[-1]):
                plt.figure(figsize=(8, 8))
                plt.imshow(landsat_raw[:, :, c], cmap='viridis')
                plt.colorbar(label='Value')
                plt.title(f'Raw Landsat Channel {c} Data')
                plt.savefig(f'landsat_raw_ch{c}_heatmap.png')
                plt.close()
                print(f"Raw Landsat channel {c} heatmap saved to landsat_raw_ch{c}_heatmap.png")
        
        # Replicate the same frame for all timestamps (Landsat updates less frequently)
        landsat_data = np.zeros((self.n_timestamps, self.dim, self.dim, 3))
        for i in range(self.n_timestamps):
            landsat_data[i] = landsat_raw
        
        np.save(cache_file, landsat_data)
        return landsat_data
    
    def _fetch_landsat_from_usgs(self):
        """Fetch Landsat data from USGS M2M API using token authentication"""
        products_available = []
        
        def send_request(url, data, api_key=None):
            json_data = json.dumps(data)
            if api_key == None:
                response = requests.post(url, json_data)
            else:
                headers = {'X-Auth-Token': api_key}              
                response = requests.post(url, json_data, headers=headers)    
            
            try:
                output = json.loads(response.text)
                if output.get('errorCode'):
                    if self.verbose:
                        print(f"API Error: {output['errorCode']} - {output.get('errorMessage', 'Unknown error')}")
                    return None
                response.close()
                return output['data']
            except Exception as e:
                response.close()
                if self.verbose:
                    print(f"Request error: {e}")
                return None
        
        service_url = "https://m2m.cr.usgs.gov/api/api/json/stable/"
        
        # Login with username and token
        payload = {'username': self.m2m_credentials['username'], 
                   'token': self.m2m_credentials['token']}
        
        if self.verbose:
            print(f"Logging in to USGS M2M API with username: {self.m2m_credentials['username']}")
        
        api_key = send_request(service_url + "login-token", payload)
        
        if not api_key:
            raise RuntimeError("Failed to login to USGS M2M API. Check credentials.")
        
        # Search for Landsat data
        dataset_name = "landsat_ard_tile_c2"
        spatial_filter = {'filterType': "mbr",
                         'lowerLeft': {'latitude': self.extent[2], 'longitude': self.extent[0]},
                         'upperRight': {'latitude': self.extent[3], 'longitude': self.extent[1]}}
        
        temporal_filter = {'start': self.start_date.strftime('%Y-%m-%d'), 
                          'end': self.end_date.strftime('%Y-%m-%d')}
        
        if self.verbose:
            print(f"Searching Landsat data for date range: {temporal_filter['start']} to {temporal_filter['end']}")
            print(f"Spatial extent: Lat [{self.extent[2]}, {self.extent[3]}], Lon [{self.extent[0]}, {self.extent[1]}]")
        
        payload = {'datasetName': dataset_name,
                   'spatialFilter': spatial_filter,
                   'temporalFilter': temporal_filter}
        
        datasets = send_request(service_url + "dataset-search", payload, api_key)
        
        if datasets:
            if self.verbose:
                print(f"Found {len(datasets)} datasets")
            
            for dataset in datasets:
                if dataset['datasetAlias'] == dataset_name:
                    acquisition_filter = {"end": self.end_date.strftime('%Y-%m-%d'),
                                        "start": self.start_date.strftime('%Y-%m-%d')}
                    
                    payload = {'datasetName': dataset['datasetAlias'], 
                              'maxResults': 2,
                              'startingNumber': 1, 
                              'sceneFilter': {'spatialFilter': spatial_filter,
                                            'acquisitionFilter': acquisition_filter}}
                    
                    scenes = send_request(service_url + "scene-search", payload, api_key)
                    
                    if scenes:
                        if self.verbose:
                            print(f"Found {scenes.get('recordsReturned', 0)} scenes")
                        
                        if scenes.get('recordsReturned', 0) > 0:
                            if self.verbose:
                                for result in scenes.get('results', []):
                                    print(f"Scene: {result.get('displayId', 'Unknown')}, Date: {result.get('acquisitionDate', 'Unknown')}")
                            
                            scene_ids = [result['entityId'] for result in scenes['results']]
                            
                            payload = {'datasetName': dataset['datasetAlias'], 
                                      'entityIds': scene_ids}
                            
                            download_options = send_request(service_url + "download-options", payload, api_key)
                            
                            if download_options:
                                downloads = []
                                for product in download_options:
                                    if product.get('available') == True:
                                        downloads.append({'entityId': product.get('entityId'),
                                                        'productId': product.get('id')})
                                
                                if downloads:
                                    label = "download-sample"
                                    payload = {'downloads': downloads, 'label': label}
                                    request_results = send_request(service_url + "download-request", payload, api_key)
                                    
                                    if request_results and 'availableDownloads' in request_results:
                                        for download in request_results['availableDownloads']:
                                            url = download.get('url', '')
                                            if url and url.split("&")[0][-2:] == "SR":
                                                products_available.append(url)
                                                if self.verbose:
                                                    print(f"Found download URL: {url}")
                    else:
                        if self.verbose:
                            print("Scene search returned no results")
        else:
            if self.verbose:
                print("Dataset search returned no results")
        
        # Logout
        send_request(service_url + "logout", None, api_key)
        
        if self.verbose:
            print(f"Found {len(products_available)} Landsat products available for download")
        
        return products_available
    
    def _process_landsat_products(self, products_available):
        """Process downloaded Landsat products"""
        landsat_data = np.zeros((self.dim, self.dim, 3))
        
        # Download and extract products
        for url in products_available:
            if self.verbose:
                print(f"Downloading Landsat product: {url}")
            
            try:
                urllib.request.urlretrieve(url, "landsat.tar")
                break  # Just get one for now
            except Exception as e:
                if self.verbose:
                    print(f"Error downloading Landsat product: {e}")
                continue
        
        if not os.path.exists("landsat.tar"):
            raise RuntimeError("Failed to download any Landsat products")
        
        # Extract specific bands
        extracted_files = []
        try:
            t = tarfile.open('landsat.tar', 'r')
            for member in t.getmembers():
                fname = member.name
                if ".TIF" in fname:
                    band = fname.split(".")[0][-5:]
                    if band == "SR_B1" or band == "SR_B7" or band == "ROSOL":
                        if self.verbose:
                            print(f"Extracting: {fname}")
                        # Create landsatHarvested directory if it doesn't exist
                        os.makedirs("landsatHarvested", exist_ok=True)
                        t.extract(member, "landsatHarvested")
                        extracted_files.append(os.path.join("landsatHarvested", fname))
            t.close()
        except Exception as e:
            if self.verbose:
                print(f"Error extracting Landsat tar file: {e}")
            if os.path.exists("landsat.tar"):
                os.remove("landsat.tar")
            raise RuntimeError(f"Error extracting Landsat tar file: {e}")
        
        if len(extracted_files) == 0:
            if os.path.exists("landsat.tar"):
                os.remove("landsat.tar")
            raise RuntimeError("No Landsat band files found in the downloaded archive")
        
        # Read and resize bands
        band_idx = 0
        for filepath in sorted(extracted_files):
            if ".TIF" in filepath:
                try:
                    if self.verbose:
                        print(f"Processing: {filepath}")
                    
                    if SKIMAGE_AVAILABLE:
                        img = io.imread(filepath)
                    elif GDAL_AVAILABLE:
                        ds = gdal.Open(filepath)
                        img = ds.ReadAsArray()
                        # Get extent for debugging
                        if self.verbose:
                            gt = ds.GetGeoTransform()
                            print(f"GeoTransform: {gt}")
                            print(f"Image size: {img.shape}")
                        ds = None  # Close dataset
                    else:
                        # Fall back to matplotlib
                        img = plt.imread(filepath)
                    
                    # Make sure img is 2D
                    if len(img.shape) > 2:
                        img = img[:,:,0]  # Take first channel
                    
                    # Add debugging for raw image data
                    if self.verbose:
                        print(f"Raw image stats for {filepath}: min={img.min()}, max={img.max()}, mean={img.mean()}")
                        
                        # Plot raw image
                        plt.figure(figsize=(10, 10))
                        plt.imshow(img, cmap='viridis')
                        plt.colorbar()
                        plt.title(f"Raw Landsat Image: {os.path.basename(filepath)}")
                        plt.savefig(f"landsat_raw_{band_idx}.png")
                        plt.close()
                        print(f"Raw image saved to landsat_raw_{band_idx}.png")
                    
                    img_resized = cv2.resize(img, (self.dim, self.dim))
                    
                    if band_idx < 3:  # Only use first 3 bands
                        landsat_data[:, :, band_idx] = img_resized
                    
                    band_idx += 1
                except Exception as e:
                    if self.verbose:
                        print(f"Error processing {filepath}: {e}")
        
        # Clean up
        if os.path.exists("landsat.tar"):
            os.remove("landsat.tar")
        if os.path.exists("landsatHarvested"):
            import shutil
            shutil.rmtree("landsatHarvested")
        
        if band_idx == 0:
            raise RuntimeError("Failed to process any Landsat bands")
        
        return landsat_data
    
    def _get_maiac_data(self):
        cache_file = os.path.join(self.cache_dir, 'maiac_data.npy')
        
        if self.use_cached_data and os.path.exists(cache_file):
            if self.verbose:
                print(f"Loading cached MAIAC data from {cache_file}")
            maiac_data = np.load(cache_file)
            
            # Add debugging info
            if self.verbose:
                print(f"MAIAC data shape: {maiac_data.shape}")
                print(f"MAIAC data stats: min={maiac_data.min():.3f}, max={maiac_data.max():.3f}, mean={maiac_data.mean():.3f}")
                
                # Check how much of the data is zero or very small
                zero_percentage = np.mean(maiac_data == 0) * 100
                small_percentage = np.mean((maiac_data > 0) & (maiac_data < 0.1)) * 100
                print(f"MAIAC zero percentage: {zero_percentage:.2f}%")
                print(f"MAIAC small values (0-0.1) percentage: {small_percentage:.2f}%")
                
                # Show histogram of values
                if np.max(maiac_data) > 0:
                    plt.figure(figsize=(10, 4))
                    plt.hist(maiac_data.flatten(), bins=50)
                    plt.title("MAIAC Data Distribution")
                    plt.xlabel("Value")
                    plt.ylabel("Count")
                    plt.yscale('log')  # Log scale for better visualization
                    plt.savefig("maiac_histogram.png")
                    print("MAIAC data histogram saved to maiac_histogram.png")
                    
                    # Also create a heatmap for MAIAC data
                    plt.figure(figsize=(10, 10))
                    plt.imshow(maiac_data[0, :, :, 0], cmap='viridis')
                    plt.colorbar()
                    plt.title("MAIAC AOD Data")
                    plt.savefig("maiac_heatmap.png")
                    print("MAIAC data heatmap saved to maiac_heatmap.png")
            
            return maiac_data
        
        if not self.maiac_credentials['username'] or not self.maiac_credentials['password']:
            raise ValueError("MAIAC credentials not found in environment variables. Set MAIAC_USERNAME and MAIAC_PASSWORD.")
        
        # Verify GDAL or PyHDF is available
        if not GDAL_AVAILABLE and not PYHDF_AVAILABLE:
            raise RuntimeError("Neither GDAL nor PyHDF is available. At least one is required for MAIAC data processing.")
        
        aod_data = self._fetch_maiac_aod()
        
        if aod_data is None:
            raise RuntimeError("Failed to fetch MAIAC AOD data from NASA.")
        
        # Debugging for raw AOD data
        if self.verbose:
            print(f"Raw AOD data shape: {aod_data.shape}")
            print(f"Raw AOD data stats: min={aod_data.min():.3f}, max={aod_data.max():.3f}, mean={aod_data.mean():.3f}")
            
            # Save raw AOD data visualization
            plt.figure(figsize=(10, 10))
            plt.imshow(aod_data, cmap='viridis')
            plt.colorbar()
            plt.title("Raw MAIAC AOD Data")
            plt.savefig("raw_maiac_aod.png")
            plt.close()
            print("Raw MAIAC AOD data visualization saved to raw_maiac_aod.png")
        
        # MAIAC data is daily, so we'll get one frame and replicate
        maiac_data = np.zeros((self.n_timestamps, self.dim, self.dim, 1))
        for i in range(self.n_timestamps):
            maiac_data[i, :, :, 0] = aod_data
        
        np.save(cache_file, maiac_data)
        return maiac_data
    
    def _fetch_maiac_aod(self):
        """Fetch MAIAC AOD data from NASA"""
        url_base = "https://e4ftl01.cr.usgs.gov/MOTA/MCD19A2.061/"
        aod_date = self.start_date.strftime("%Y.%m.%d")
        url = url_base + aod_date
        ext = 'hdf'
        
        if self.verbose:
            print(f"Looking for MAIAC data on date: {aod_date}")
            print(f"URL: {url}")
        
        def listFD(url, ext=''):
            try:
                page = requests.get(url).text
                soup = BeautifulSoup(page, 'html.parser')
                files = [url + '/' + node.get('href') for node in soup.find_all('a') 
                       if node.get('href').endswith(ext) and "h08v05" in node.get('href')]
                if self.verbose:
                    print(f"Found {len(files)} HDF files")
                return files
            except Exception as e:
                if self.verbose:
                    print(f"Error listing files: {e}")
                return []
        
        files = listFD(url, ext)
        if not files:
            if self.verbose:
                print(f"No MAIAC files found for date {aod_date}, trying alternative date")
            
            # Try previous day
            alt_date = (self.start_date - timedelta(days=1)).strftime("%Y.%m.%d")
            alt_url = url_base + alt_date
            files = listFD(alt_url, ext)
            
            if not files and self.verbose:
                print(f"No MAIAC files found for alternative date {alt_date} either")
                raise RuntimeError(f"No MAIAC files found for date {aod_date} or alternative")
        
        if not files:
            raise RuntimeError(f"No MAIAC files found for date {aod_date}")
        
        file_url = files[0]
        if self.verbose:
            print(f"Attempting to download: {file_url}")
        
        output_file = 'aod.hdf'
        
        try:
            with requests.Session() as session:
                session.auth = (self.maiac_credentials['username'], 
                               self.maiac_credentials['password'])
                r1 = session.request('get', file_url)
                if self.verbose:
                    print(f"Initial request status: {r1.status_code}")
                
                r = session.get(r1.url, auth=(self.maiac_credentials['username'], 
                                             self.maiac_credentials['password']))
                
                if not r.ok:
                    raise RuntimeError(f"Failed to download MAIAC file. Status code: {r.status_code}")
                
                if self.verbose:
                    print(f"Download successful, file size: {len(r.content)} bytes")
                
                with open(output_file, 'wb') as f:
                    for data in r.iter_content(chunk_size=8192):
                        f.write(data)
                
                # Verify file exists and has content
                if os.path.exists(output_file):
                    file_size = os.path.getsize(output_file)
                    if self.verbose:
                        print(f"{output_file} saved, size: {file_size} bytes")
                    if file_size == 0:
                        raise RuntimeError(f"Downloaded file {output_file} is empty")
                else:
                    raise RuntimeError(f"Failed to save {output_file}")
                
                # Process the HDF file
                try:
                    aod_data = self._process_hdf_file(output_file)
                    
                    # Clean up
                    os.remove(output_file)
                    
                    return aod_data
                except Exception as e:
                    # Clean up before re-raising
                    if os.path.exists(output_file):
                        os.remove(output_file)
                    raise RuntimeError(f"Error processing HDF file: {e}")
                
        except Exception as e:
            if os.path.exists(output_file):
                os.remove(output_file)
            raise RuntimeError(f"Error in MAIAC download process: {e}")
    
    def _process_hdf_file(self, hdf_file):
        """Process HDF file using available methods"""
        # Try GDAL first if available
        if GDAL_AVAILABLE:
            try:
                if self.verbose:
                    print("Processing HDF file with GDAL")
                
                # Open HDF file with GDAL
                gdal.UseExceptions()  # Enable exceptions to catch errors
                hdf_ds = gdal.Open(hdf_file)
                
                if hdf_ds is None:
                    raise RuntimeError("GDAL could not open HDF file")
                
                # List subdatasets
                subdatasets = hdf_ds.GetSubDatasets()
                if self.verbose:
                    print(f"Found {len(subdatasets)} subdatasets")
                    for i, subds in enumerate(subdatasets):
                        print(f"  Subdataset {i}: {subds[0]} - {subds[1]}")
                
                if len(subdatasets) == 0:
                    raise RuntimeError("No subdatasets found in HDF file")
                
                # Look for AOD dataset
                aod_ds = None
                aod_subds_name = None
                for subds in subdatasets:
                    if 'Optical_Depth_055' in subds[0] or 'AOD' in subds[0]:
                        aod_subds_name = subds[0]
                        aod_ds = gdal.Open(subds[0])
                        if self.verbose:
                            print(f"Found AOD dataset: {subds[0]}")
                        break
                
                if aod_ds is None:
                    # Use first subdataset as fallback
                    aod_subds_name = subdatasets[0][0]
                    aod_ds = gdal.Open(subdatasets[0][0])
                    if self.verbose:
                        print(f"Using first subdataset as fallback: {subdatasets[0][0]}")
                
                # Get dimensions and validate
                width = aod_ds.RasterXSize
                height = aod_ds.RasterYSize
                bands = aod_ds.RasterCount
                
                if self.verbose:
                    print(f"Dataset dimensions: {width}x{height}")
                    print(f"Number of bands: {bands}")
                    
                    # Get metadata
                    metadata = aod_ds.GetMetadata()
                    if metadata:
                        print("Metadata:")
                        for key, value in metadata.items():
                            print(f"  {key}: {value}")
                
                if width <= 0 or height <= 0:
                    raise RuntimeError(f"Invalid dataset dimensions: {width}x{height}")
                
                # Read array
                aod_array = aod_ds.ReadAsArray()
                
                # Check if array is valid
                if aod_array is None or aod_array.size == 0:
                    raise RuntimeError("Read array is empty or None")
                    
                if self.verbose:
                    print(f"AOD array shape: {aod_array.shape}, dtype: {aod_array.dtype}")
                    
                    # Output a histogram of values
                    plt.figure(figsize=(10, 4))
                    plt.hist(aod_array.flatten(), bins=50)
                    plt.title(f"Raw AOD Data Distribution - {os.path.basename(aod_subds_name)}")
                    plt.xlabel("Value")
                    plt.ylabel("Count")
                    plt.yscale('log')  # Log scale for better visualization
                    plt.savefig("raw_aod_histogram.png")
                    plt.close()
                    print("Raw AOD histogram saved to raw_aod_histogram.png")
                
                # Handle multi-band arrays (3D arrays)
                # If we have a 3D array (bands, height, width), use only the first band
                if len(aod_array.shape) == 3:
                    if self.verbose:
                        print(f"Multi-band array detected. Using band 0 (shape: {aod_array[0].shape})")
                        # Visualize all bands
                        for b in range(aod_array.shape[0]):
                            plt.figure(figsize=(8, 8))
                            plt.imshow(aod_array[b], cmap='viridis')
                            plt.colorbar()
                            plt.title(f"AOD Band {b}")
                            plt.savefig(f"aod_band_{b}.png")
                            plt.close()
                            print(f"Band {b} visualization saved to aod_band_{b}.png")
                    
                    aod_array = aod_array[0]  # Take the first band
                
                # Ensure array has proper dimensions before resize
                if len(aod_array.shape) != 2 or aod_array.shape[0] <= 0 or aod_array.shape[1] <= 0:
                    raise RuntimeError(f"Invalid array dimensions for resize: {aod_array.shape}")
                
                # Scale the data if needed
                # Check for negative fill values or very large values that might need scaling
                if np.min(aod_array) < 0 or np.max(aod_array) > 10000:
                    if self.verbose:
                        print(f"Scaling AOD data. Original range: [{np.min(aod_array)}, {np.max(aod_array)}]")
                    
                    # Identify fill values and mask them
                    if np.min(aod_array) < 0:
                        mask = aod_array < 0
                        aod_array[mask] = 0
                        if self.verbose:
                            print(f"Masked {np.sum(mask)} negative values")
                    
                    # Scale large values if needed
                    if np.max(aod_array) > 10000:
                        scale_factor = 0.001  # Common scale factor for AOD
                        aod_array = aod_array * scale_factor
                        if self.verbose:
                            print(f"Applied scale factor {scale_factor}. New range: [{np.min(aod_array)}, {np.max(aod_array)}]")
                
                # Close datasets
                aod_ds = None
                hdf_ds = None
                
                # Explicitly set dimensions for resize
                target_size = (self.dim, self.dim)
                
                # Resize with explicit dimensions and ensure inputs are valid
                if target_size[0] > 0 and target_size[1] > 0 and aod_array.size > 0:
                    try:
                        # Try first with OpenCV resize
                        aod_resized = cv2.resize(aod_array, target_size, interpolation=cv2.INTER_LINEAR)
                        if self.verbose:
                            print(f"Resized with OpenCV. New shape: {aod_resized.shape}")
                    except Exception as resize_error:
                        if self.verbose:
                            print(f"OpenCV resize failed: {resize_error}, trying alternative method")
                        
                        # Alternative resize using scipy for more robust handling
                        from scipy.ndimage import zoom
                        
                        # Calculate zoom factors (same for both dimensions)
                        zoom_factors = (self.dim / aod_array.shape[0], self.dim / aod_array.shape[1])
                        
                        # Apply zoom (bilinear interpolation)
                        aod_resized = zoom(aod_array, zoom_factors, order=1)
                        
                        if self.verbose:
                            print(f"Resized with scipy.ndimage.zoom. New shape: {aod_resized.shape}")
                else:
                    raise RuntimeError(f"Invalid resize dimensions or input array: target={target_size}, array.shape={aod_array.shape}")
                
                # Final visualization of the resized data
                if self.verbose:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(aod_resized, cmap='viridis')
                    plt.colorbar()
                    plt.title("Resized AOD Data")
                    plt.savefig("resized_aod.png")
                    plt.close()
                    print("Resized AOD visualization saved to resized_aod.png")
                
                return aod_resized
                
            except Exception as e:
                if self.verbose:
                    print(f"GDAL processing failed: {e}")
                # If GDAL fails, try PyHDF
                if not PYHDF_AVAILABLE:
                    raise RuntimeError(f"GDAL failed to process HDF file and PyHDF not available: {e}")
        
        # Try pyhdf if GDAL failed or not available
        if PYHDF_AVAILABLE:
            try:
                if self.verbose:
                    print("Processing HDF file with pyhdf")
                
                # Open HDF file with pyhdf
                hdf = SD(hdf_file, SDC.READ)
                
                # List datasets
                datasets = hdf.datasets()
                if self.verbose:
                    print(f"Available datasets: {list(datasets.keys())}")
                
                # Look for AOD dataset
                aod_name = None
                for name in datasets.keys():
                    if 'Optical_Depth_055' in name or 'AOD' in name:
                        aod_name = name
                        break
                
                if aod_name is None:
                    # Use first dataset as fallback
                    aod_name = list(datasets.keys())[0]
                    if self.verbose:
                        print(f"Using first dataset as fallback: {aod_name}")
                
                if self.verbose:
                    print(f"Using dataset: {aod_name}")
                
                # Get AOD data
                aod_sds = hdf.select(aod_name)
                aod_data = aod_sds.get()
                
                # Get attributes
                if self.verbose:
                    attrs = aod_sds.attributes()
                    print("Dataset attributes:")
                    for key, value in attrs.items():
                        print(f"  {key}: {value}")
                
                # Close file
                aod_sds.endsel()
                hdf.end()
                
                # Check if array is valid
                if aod_data is None or aod_data.size == 0:
                    raise RuntimeError("PyHDF: Read array is empty or None")
                
                if self.verbose:
                    print(f"PyHDF AOD array shape: {aod_data.shape}, dtype: {aod_data.dtype}")
                    
                    # Output a histogram of values
                    plt.figure(figsize=(10, 4))
                    plt.hist(aod_data.flatten(), bins=50)
                    plt.title(f"Raw AOD Data Distribution (PyHDF) - {aod_name}")
                    plt.xlabel("Value")
                    plt.ylabel("Count")
                    plt.yscale('log')  # Log scale for better visualization
                    plt.savefig("raw_aod_pyhdf_histogram.png")
                    plt.close()
                    print("Raw AOD PyHDF histogram saved to raw_aod_pyhdf_histogram.png")
                
                # Handle multi-band arrays (3D arrays)
                if len(aod_data.shape) == 3:
                    if self.verbose:
                        print(f"Multi-band array detected. Using band 0")
                    aod_data = aod_data[0]  # Take the first band
                
                # Scale the data if needed
                # Check for negative fill values or very large values that might need scaling
                if np.min(aod_data) < 0 or np.max(aod_data) > 10000:
                    if self.verbose:
                        print(f"Scaling AOD data. Original range: [{np.min(aod_data)}, {np.max(aod_data)}]")
                    
                    # Identify fill values and mask them
                    if np.min(aod_data) < 0:
                        mask = aod_data < 0
                        aod_data[mask] = 0
                        if self.verbose:
                            print(f"Masked {np.sum(mask)} negative values")
                    
                    # Scale large values if needed
                    if np.max(aod_data) > 10000:
                        scale_factor = 0.001  # Common scale factor for AOD
                        aod_data = aod_data * scale_factor
                        if self.verbose:
                            print(f"Applied scale factor {scale_factor}. New range: [{np.min(aod_data)}, {np.max(aod_data)}]")
                
                # Resize with explicit dimensions and validation
                if self.dim > 0 and aod_data.size > 0:
                    try:
                        # Resize with OpenCV
                        aod_resized = cv2.resize(aod_data, (self.dim, self.dim), interpolation=cv2.INTER_LINEAR)
                    except Exception as resize_error:
                        if self.verbose:
                            print(f"PyHDF OpenCV resize failed: {resize_error}, trying alternative method")
                        
                        # Alternative resize using scipy
                        from scipy.ndimage import zoom
                        
                        # Calculate zoom factors
                        zoom_factors = (self.dim / aod_data.shape[0], self.dim / aod_data.shape[1])
                        
                        # Apply zoom (bilinear interpolation)
                        aod_resized = zoom(aod_data, zoom_factors, order=1)
                else:
                    raise RuntimeError(f"PyHDF: Invalid resize dimensions or input array: target={self.dim}, array.shape={aod_data.shape}")
                
                # Final visualization of the resized data
                if self.verbose:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(aod_resized, cmap='viridis')
                    plt.colorbar()
                    plt.title("Resized AOD Data (PyHDF)")
                    plt.savefig("resized_aod_pyhdf.png")
                    plt.close()
                    print("Resized AOD PyHDF visualization saved to resized_aod_pyhdf.png")
                
                return aod_resized
                
            except Exception as e:
                raise RuntimeError(f"pyhdf processing failed: {e}")
        
        # If we get here with no data, raise an error
        raise RuntimeError("Could not process HDF file with any available method")
    
    def _get_wind_data(self):
        # Only run this if include_wind is True
        if not self.include_wind:
            if self.verbose:
                print("Wind data processing skipped (include_wind=False)")
            return None
            
        cache_file = os.path.join(self.cache_dir, 'wind_data.npy')
        
        if self.use_cached_data and os.path.exists(cache_file):
            if self.verbose:
                print(f"Loading cached wind data from {cache_file}")
            wind_data = np.load(cache_file)
            
            # Add debugging info
            if self.verbose:
                print(f"Wind data shape: {wind_data.shape}")
                print(f"Wind data stats: min={wind_data.min():.3f}, max={wind_data.max():.3f}, mean={wind_data.mean():.3f}")
                
                # Visualize wind data
                plt.figure(figsize=(10, 10))
                plt.imshow(wind_data[0, :, :, 0], cmap='viridis')
                plt.colorbar(label='Wind Speed (m/s)')
                plt.title('Wind Speed Data')
                plt.savefig('wind_data_visualization.png')
                plt.close()
                print("Wind data visualization saved to wind_data_visualization.png")
            
            return wind_data
        
        if not self.openweather_api_key:
            raise ValueError("OpenWeather API key not found in environment variables. Set OPENWEATHER_API_KEY.")
        
        wind_data = np.zeros((self.n_timestamps, self.dim, self.dim, 1))
        
        # Reduce frequency of sampling to minimize API calls
        sample_freq = 24  # One sample per day
        sampled_timestamps = self.timestamps[::sample_freq]
        
        for i, timestamp in enumerate(sampled_timestamps):
            if self.verbose:
                print(f"Fetching wind data for day {i+1}/{len(sampled_timestamps)}")
            
            try:
                wind_frame = self._fetch_wind_data()
                
                # Fill multiple timestamps with the same frame to reduce API calls
                start_idx = i * sample_freq
                end_idx = min(start_idx + sample_freq, self.n_timestamps)
                
                for j in range(start_idx, end_idx):
                    wind_data[j, :, :, 0] = wind_frame
                    
            except Exception as e:
                if self.verbose:
                    print(f"Failed to fetch wind data for day {i+1}: {e}")
                # Continue with zeros for this day
        
        np.save(cache_file, wind_data)
        return wind_data
    
    def _fetch_wind_data(self):
        """Fetch wind data from OpenWeatherMap API"""
        # Only run this if include_wind is True
        if not self.include_wind:
            return None
            
        lon_min, lon_max, lat_min, lat_max = self.extent
        
        # Reduce sample size to minimize API calls
        sample_size = 5  # Reduced from 10
        lat_points = np.linspace(lat_min, lat_max, sample_size)
        lon_points = np.linspace(lon_min, lon_max, sample_size)
        
        sampled_values = np.zeros((sample_size, sample_size))
        fetch_failed = True
        
        for i, lat in enumerate(lat_points):
            for j, lon in enumerate(lon_points):
                url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.openweather_api_key}"
                
                try:
                    response = requests.get(url)
                    if not response.ok:
                        if self.verbose:
                            print(f"OpenWeatherMap API error: {response.status_code}")
                        continue
                    
                    data = response.json()
                    if 'wind' not in data or 'speed' not in data['wind']:
                        if self.verbose:
                            print(f"Invalid wind data structure: {data}")
                        continue
                    
                    sampled_values[i, j] = data['wind']['speed']
                    fetch_failed = False
                except Exception as e:
                    if self.verbose:
                        print(f"Error fetching wind data for point ({lat}, {lon}): {e}")
                
                # Rate limiting - increase to reduce chance of being rate-limited
                time.sleep(0.3)
        
        # Check if data fetch was successful
        if fetch_failed:
            raise RuntimeError("Failed to fetch wind data from OpenWeatherMap API.")
        
        # Debug visualization of raw sampled values
        if self.verbose:
            plt.figure(figsize=(8, 8))
            plt.imshow(sampled_values, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Wind Speed (m/s)')
            plt.title('Raw Wind Speed Data (Before Interpolation)')
            plt.savefig('raw_wind_samples.png')
            print("Raw wind samples saved to raw_wind_samples.png")
        
        # Interpolate to full grid
        wind_grid = cv2.resize(sampled_values, (self.dim, self.dim), interpolation=cv2.INTER_LINEAR)
        
        # Debug visualization of interpolated grid
        if self.verbose:
            plt.figure(figsize=(10, 10))
            plt.imshow(wind_grid, cmap='viridis')
            plt.colorbar(label='Wind Speed (m/s)')
            plt.title('Interpolated Wind Speed Data')
            plt.savefig('interpolated_wind_grid.png')
            print("Interpolated wind grid saved to interpolated_wind_grid.png")
        
        return wind_grid
    
    def _normalize_channels(self, channels):
        """Normalize each channel using mean and standard deviation"""
        normalized = {}
        
        for name, data in channels.items():
            if self.verbose:
                print(f"\nNormalizing {name} data:")
                print(f"  Before normalization: min={data.min():.3f}, max={data.max():.3f}, mean={data.mean():.3f}")
            
            normalized_data = data.copy()
            
            for c in range(data.shape[-1]):
                channel_data = data[:, :, :, c]
                
                # Special handling for MAIAC data to fix the negative fill values
                if name == 'maiac':
                    # Mask out extreme negative values that are likely fill values
                    valid_mask = channel_data > -1000  # Adjust threshold as needed
                    
                    # Calculate statistics only on valid data
                    if np.any(valid_mask):
                        valid_data = channel_data[valid_mask]
                        mean = valid_data.mean()
                        std = valid_data.std()
                        
                        # Replace fill values with zeros before normalization
                        masked_data = channel_data.copy()
                        masked_data[~valid_mask] = 0
                        
                        if self.verbose:
                            print(f"  MAIAC Channel {c}: Masked {np.sum(~valid_mask)} likely fill values")
                            print(f"  Valid data stats: min={valid_data.min():.3f}, max={valid_data.max():.3f}, mean={mean:.3f}, std={std:.3f}")
                        
                        # Normalize using statistics from valid data only
                        if std > 0:
                            normalized_data[:, :, :, c] = (masked_data - mean) / std
                        else:
                            normalized_data[:, :, :, c] = 0
                    else:
                        # If no valid data, set to zeros
                        normalized_data[:, :, :, c] = 0
                        if self.verbose:
                            print(f"  WARNING: No valid MAIAC data found in channel {c}")
                else:
                    # Standard normalization for non-MAIAC data
                    mean = channel_data.mean()
                    std = channel_data.std()
                    
                    if self.verbose:
                        print(f"  Channel {c}: mean={mean:.3f}, std={std:.3f}")
                    
                    if std > 0:
                        normalized_data[:, :, :, c] = (channel_data - mean) / std
                    else:
                        normalized_data[:, :, :, c] = 0
                    
                if self.verbose:
                    normalized_channel = normalized_data[:, :, :, c]
                    print(f"    After: min={normalized_channel.min():.3f}, max={normalized_channel.max():.3f}, mean={normalized_channel.mean():.3f}")
            
            if self.verbose:
                print(f"  After normalization: min={normalized_data.min():.3f}, max={normalized_data.max():.3f}, mean={normalized_data.mean():.3f}")
                
                # Visualize before/after normalization for first frame
                for c in range(data.shape[-1]):
                    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Before normalization
                    im0 = axes[0].imshow(data[0, :, :, c], cmap='viridis')
                    axes[0].set_title(f"{name} Ch{c} Before Normalization")
                    plt.colorbar(im0, ax=axes[0])
                    
                    # After normalization
                    im1 = axes[1].imshow(normalized_data[0, :, :, c], cmap='viridis')
                    axes[1].set_title(f"{name} Ch{c} After Normalization")
                    plt.colorbar(im1, ax=axes[1])
                    
                    plt.tight_layout()
                    plt.savefig(f"{name}_ch{c}_normalization.png")
                    plt.close()
                    print(f"Normalization visualization saved to {name}_ch{c}_normalization.png")
            
            normalized[name] = normalized_data
        
        return normalized
    
    def _sliding_window_of(self, frames, window_size):
        """Create sliding window samples"""
        n_frames, row, col, channels = frames.shape
        n_samples = n_frames - window_size + 1
        
        if n_samples <= 0:
            raise ValueError(f"Not enough frames ({n_frames}) for sliding window of size {window_size}")
        
        samples = np.empty((n_samples, window_size, row, col, channels))
        
        for i in range(n_samples):
            samples[i] = frames[i:i+window_size]
        
        return samples
    
    def get_channel_info(self):
        # Base channels (always included)
        channel_order = [
            'Landsat_SR_B1',       # 0
            'Landsat_SR_B7',       # 1
            'Landsat_QA_AEROSOL',  # 2
            'MAIAC_AOD',           # 3
        ]
        
        # Add wind if available
        if hasattr(self, 'has_wind_data') and self.has_wind_data:
            channel_order.append('Wind_Speed')  # 4
        
        return {
            'landsat_bands': ['SR_B1', 'SR_B7', 'SR_QA_AEROSOL'],
            'maiac_bands': ['AOD_550nm'],
            'wind_bands': ['Wind_Speed'] if hasattr(self, 'has_wind_data') and self.has_wind_data else [],
            'total_channels': len(channel_order),
            'channel_order': channel_order
        }
    
    def visualize_channels(self, sample_idx=0, frame_idx=0):
        if sample_idx >= len(self.data):
            print(f"Sample index {sample_idx} out of range. Using 0.")
            sample_idx = 0
        
        if frame_idx >= self.frames_per_sample:
            print(f"Frame index {frame_idx} out of range. Using 0.")
            frame_idx = 0
        
        sample = self.data[sample_idx, frame_idx]
        channel_info = self.get_channel_info()
        
        num_channels = sample.shape[-1]
        
        # Calculate subplot grid dimensions
        cols = min(3, num_channels)
        rows = (num_channels + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*3.5))
        
        # Handle case of a single subplot
        if num_channels == 1:
            axes = np.array([axes])
        
        # Convert to 1D array for easier indexing
        axes = np.array(axes).ravel()
        
        for i, channel_name in enumerate(channel_info['channel_order']):
            if i < num_channels:
                ax = axes[i]
                channel_data = sample[:, :, i]
                
                # Calculate appropriate min/max for visualization
                # This ensures we're not using identical values for vmin and vmax
                vmin = np.percentile(channel_data, 1)  # 1st percentile to avoid extremes
                vmax = np.percentile(channel_data, 99)  # 99th percentile to avoid extremes
                
                # Make sure vmin and vmax are not identical
                if vmin == vmax:
                    # If they're identical, create a small range
                    vmin = vmin - 0.1 * abs(vmin) if vmin != 0 else -0.1
                    vmax = vmax + 0.1 * abs(vmax) if vmax != 0 else 0.1
                
                im = ax.imshow(channel_data, cmap='viridis', vmin=vmin, vmax=vmax)
                ax.set_title(f"{channel_name}\nmin={channel_data.min():.2f}, max={channel_data.max():.2f}")
                ax.axis('off')
                plt.colorbar(im, ax=ax)
        
        # Hide unused subplots
        for i in range(num_channels, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'PWWB Channels - Sample {sample_idx}, Frame {frame_idx}')
        plt.tight_layout()
        plt.show()

    def visualize_combined_data(self, data, channel_names):
        """Visualize all channels in the combined data."""
        if self.verbose:
            print("5. Visualizing combined data...")
        
        num_channels = data.shape[-1]
        
        # Calculate subplot grid dimensions
        cols = min(3, num_channels)
        rows = (num_channels + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
        
        # Handle case of a single subplot
        if num_channels == 1:
            axes = np.array([axes])
        
        # Convert to 1D array for easier indexing
        axes = np.array(axes).ravel()
        
        for i, channel_name in enumerate(channel_names):
            if i < num_channels:
                ax = axes[i]
                channel_data = data[0, 0, :, :, i]
                
                # Calculate appropriate min/max for visualization
                if "MAIAC" in channel_name:
                    # Special handling for MAIAC data
                    # Use percentiles to avoid outliers
                    vmin = np.min(channel_data)
                    vmax = np.percentile(channel_data, 99)
                    
                    # Ensure vmin and vmax are different
                    if abs(vmax - vmin) < 0.1:
                        # If too small difference or same values, create artificial range
                        vmin = -0.5
                        vmax = 0.5
                else:
                    # For other channels
                    vmin = np.percentile(channel_data, 1)
                    vmax = np.percentile(channel_data, 99)
                    
                    # Ensure vmin and vmax are different
                    if abs(vmax - vmin) < 0.001:
                        # If too small difference or same values, create artificial range
                        vmin = np.min(channel_data) - 0.1 * abs(np.min(channel_data)) if np.min(channel_data) != 0 else -0.1
                        vmax = np.max(channel_data) + 0.1 * abs(np.max(channel_data)) if np.max(channel_data) != 0 else 0.1
                
                if self.verbose:
                    print(f"  Channel {i} ({channel_name}): min={channel_data.min():.3f}, max={channel_data.max():.3f}, mean={channel_data.mean():.3f}")
                    print(f"    Using display range: vmin={vmin:.3f}, vmax={vmax:.3f}")
                
                im = ax.imshow(channel_data, cmap='viridis', vmin=vmin, vmax=vmax)
                ax.set_title(f"Channel {i}: {channel_name}")
                plt.colorbar(im, ax=ax)
                ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_channels, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('combined_channels_visualization.png')
        
        if self.verbose:
            print(f"✓ Visualization saved to 'combined_channels_visualization.png'")
    
    def save_data(self, filepath):
        np.save(filepath, self.data)
        if self.verbose:
            print(f"Data saved to {filepath}")
    
    def load_data(self, filepath):
        self.data = np.load(filepath)
        if self.verbose:
            print(f"Data loaded from {filepath}")
            print(f"Data shape: {self.data.shape}")