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
    from skimage.color import rgb2gray
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: skimage.io not available. Will use alternative methods for image reading.")

class PWWBData:
    """
    PWWBData class for processing satellite imagery data for machine learning.
    
    This class handles:
    - Landsat 8 data (3 bands): SR_B1, SR_B7, and SR_QA_AEROSOL
    - MAIAC AOD data (1 band)
    
    Following the approach in pm25Script_noGCN.py, this implementation:
    - Uses a single MAIAC observation from the start date for all timestamps
    - Applies proper scaling to get physical AOD values
    - Preserves data in physical units for standardization in ML pipeline
    """
    def __init__(
        self,
        start_date,
        end_date,
        extent=(-118.4, -118.0, 33.9, 34.2),  # LA area bounds from experiment template
        frames_per_sample=5,
        dim=200,
        cache_dir='data/pwwb_cache/',
        use_cached_data=True,
        verbose=False,
        env_file='.env',
        output_dir="visualization_output"
    ):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.extent = extent
        self.frames_per_sample = frames_per_sample
        self.dim = dim
        self.verbose = verbose
        self.cache_dir = cache_dir
        self.use_cached_data = use_cached_data
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
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
        
        # Generate timestamps at hourly intervals
        self.timestamps = pd.date_range(self.start_date, self.end_date, freq='H')
        self.n_timestamps = len(self.timestamps)
        
        if self.verbose:
            print(f"Initialized PWWBData with {self.n_timestamps} hourly timestamps")
            print(f"Date range: {self.start_date} to {self.end_date}")
            print(f"Note: Following pattern in pm25Script_noGCN.py, MAIAC data will use a single observation from start date ({self.start_date.strftime('%Y-%m-%d')})")
        
        self._process_pipeline()
    
    def _process_pipeline(self):
        """Main processing pipeline"""
        channels = {}
        
        if self.verbose:
            print("Processing Landsat data...")
        channels['landsat'] = self._get_landsat_data()
        
        if self.verbose:
            print("Processing MAIAC AOD data (from start date only)...")
        channels['maiac'] = self._get_maiac_data()
        
        # Concatenate raw channels - no normalization
        channel_list = [
            channels['landsat'],  # 3 channels
            channels['maiac'],    # 1 channel
        ]
        
        self.all_channels = np.concatenate(channel_list, axis=-1)
        
        # Visualize combined channels
        if self.verbose:
            channel_names = self.get_channel_info()['channel_order']
            self.visualize_combined_data(
                np.expand_dims(np.expand_dims(self.all_channels, 0), 0),  # Add batch and time dimensions
                channel_names
            )
        
        self.data = self._sliding_window_of(self.all_channels, self.frames_per_sample)
        
        # Store individual channels for access
        self.landsat_data = channels['landsat']
        self.maiac_data = channels['maiac']
        
        if self.verbose:
            print(f"Final data shape: {self.data.shape}")
            self._print_data_statistics()
    
    def _get_landsat_data(self):
        """Get Landsat data from USGS API or cache"""
        cache_file = os.path.join(self.cache_dir, 'landsat_data.npy')
        
        if self.use_cached_data and os.path.exists(cache_file):
            if self.verbose:
                print(f"Loading cached Landsat data from {cache_file}")
            return np.load(cache_file)
        
        if not self.m2m_credentials['username'] or not self.m2m_credentials['token']:
            raise ValueError("M2M credentials not found in environment variables. Set M2M_USERNAME and M2M_TOKEN.")
        
        products_available = self._fetch_landsat_from_usgs()
        if not products_available:
            raise RuntimeError("No Landsat products available for the specified date range and extent.")
        
        landsat_raw = self._process_landsat_products(products_available)
        
        # Replicate the same frame for all timestamps (Landsat updates less frequently)
        landsat_data = np.zeros((self.n_timestamps, self.dim, self.dim, 3))
        for i in range(self.n_timestamps):
            landsat_data[i] = landsat_raw
        
        np.save(cache_file, landsat_data)
        return landsat_data
    
    def _fetch_landsat_from_usgs(self):
        """Fetch Landsat data from USGS M2M API"""
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
                    
                    if scenes and scenes.get('recordsReturned', 0) > 0:
                        if self.verbose:
                            print(f"Found {scenes.get('recordsReturned', 0)} scenes")
                            
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
        
        # Logout
        send_request(service_url + "logout", None, api_key)
        
        if self.verbose:
            print(f"Found {len(products_available)} Landsat products available for download")
        
        return products_available
    
    def _process_landsat_products(self, products_available):
        """Process downloaded Landsat products"""
        landsat_data = np.zeros((self.dim, self.dim, 3))
        
        # Create landsatHarvested directory if it doesn't exist
        os.makedirs("landsatHarvested", exist_ok=True)
        
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
                    
                    # Use plt.imread to read TIF files
                    img = plt.imread(filepath)
                    
                    # Make sure img is 2D
                    if len(img.shape) > 2:
                        img = img[:,:,0]  # Take first channel
                    
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
        """
        Get MAIAC AOD data from NASA or cache.
        
        Following the approach in pm25Script_noGCN.py, this method only retrieves
        MAIAC data for the start date and uses it for all timestamps in the range.
        This approach is appropriate because:
        1. MAIAC data is only updated 1-2 times per day at most for a location
        2. Cloud cover can result in many days without valid observations
        3. AOD patterns generally change more slowly than other variables
        """
        cache_file = os.path.join(self.cache_dir, 'maiac_data.npy')
        
        if self.use_cached_data and os.path.exists(cache_file):
            if self.verbose:
                print(f"Loading cached MAIAC data from {cache_file}")
            return np.load(cache_file)
        
        if not self.maiac_credentials['username'] or not self.maiac_credentials['password']:
            raise ValueError("MAIAC credentials not found in environment variables. Set MAIAC_USERNAME and MAIAC_PASSWORD.")
        
        # Verify data processing capabilities
        if not GDAL_AVAILABLE and not PYHDF_AVAILABLE and not SKIMAGE_AVAILABLE:
            raise RuntimeError("Neither GDAL, pyhdf, nor skimage.io is available. At least one is required for MAIAC data processing.")
        
        # Get MAIAC data for the start date only (following pm25Script_noGCN.py)
        if self.verbose:
            print(f"Retrieving MAIAC data for start date: {self.start_date.strftime('%Y-%m-%d')}")
        
        aod_data = self._fetch_maiac_aod(self.start_date)
        
        if aod_data is None:
            # Try the next day if the start date has no data
            alt_date = self.start_date + timedelta(days=1)
            if self.verbose:
                print(f"No MAIAC data for start date, trying next day: {alt_date.strftime('%Y-%m-%d')}")
            aod_data = self._fetch_maiac_aod(alt_date)
            
            # Try previous day if still no data
            if aod_data is None:
                alt_date = self.start_date - timedelta(days=1)
                if self.verbose:
                    print(f"Still no MAIAC data, trying previous day: {alt_date.strftime('%Y-%m-%d')}")
                aod_data = self._fetch_maiac_aod(alt_date)
        
        if aod_data is None:
            raise RuntimeError("Failed to fetch MAIAC AOD data for the start date or adjacent days.")
        
        # Create a 4D array with the same MAIAC data for all timestamps (n_timestamps, height, width, 1)
        maiac_data = np.zeros((self.n_timestamps, self.dim, self.dim, 1))
        for i in range(self.n_timestamps):
            maiac_data[i, :, :, 0] = aod_data
        
        if self.verbose:
            print(f"Successfully retrieved MAIAC data. Replicating for all {self.n_timestamps} timestamps.")
            # Calculate statistics on the AOD data
            non_zero = np.count_nonzero(aod_data)
            total = aod_data.size
            coverage = (non_zero / total) * 100
            print(f"MAIAC data coverage: {coverage:.2f}% ({non_zero}/{total} pixels)")
            print(f"MAIAC value range: {np.min(aod_data):.5f} to {np.max(aod_data):.5f}")
            print(f"MAIAC mean: {np.mean(aod_data):.5f}")
        
        np.save(cache_file, maiac_data)
        return maiac_data
    
    def _fetch_maiac_aod(self, date):
        """
        Fetch MAIAC AOD data from NASA for a specific date
        
        Parameters:
        -----------
        date : datetime
            The specific date to fetch MAIAC data for
        
        Returns:
        --------
        aod_data : numpy.ndarray
            The processed AOD data for the specified date, or None if no data available
        """
        url_base = "https://e4ftl01.cr.usgs.gov/MOTA/MCD19A2.061/"
        aod_date = date.strftime("%Y.%m.%d")
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
                print(f"No MAIAC files found for date {aod_date}")
            return None
        
        file_url = files[0]
        if self.verbose:
            print(f"Attempting to download: {file_url}")
        
        # Create a temporary directory for the MAIAC file
        temp_dir = "maiac_temp"
        os.makedirs(temp_dir, exist_ok=True)
        output_file = os.path.join(temp_dir, f'aod_{date.strftime("%Y%m%d")}.hdf')
        
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
                    if self.verbose:
                        print(f"Failed to download MAIAC file. Status code: {r.status_code}")
                    return None
                
                if self.verbose:
                    print(f"Download successful, file size: {len(r.content)} bytes")
                
                with open(output_file, 'wb') as f:
                    for data in r.iter_content(chunk_size=8192):
                        f.write(data)
                
                # Process the HDF file
                try:
                    if GDAL_AVAILABLE:
                        aod_data = self._process_hdf_with_gdal(output_file)
                        if aod_data is not None:
                            return aod_data
                    
                    if PYHDF_AVAILABLE:
                        aod_data = self._process_hdf_with_pyhdf(output_file)
                        if aod_data is not None:
                            return aod_data
                    
                    if SKIMAGE_AVAILABLE:
                        aod_data = self._process_hdf_with_skimage(output_file)
                        if aod_data is not None:
                            return aod_data
                    
                    # If we reach here, all methods failed
                    if self.verbose:
                        print("All methods failed to process MAIAC data")
                    return None
                
                except Exception as e:
                    # Clean up before re-raising
                    if self.verbose:
                        print(f"Error processing HDF file: {e}")
                    return None
                finally:
                    # Clean up after processing
                    if os.path.exists(output_file):
                        os.remove(output_file)
            
        except Exception as e:
            if self.verbose:
                print(f"Error in MAIAC download process: {e}")
            if os.path.exists(output_file):
                os.remove(output_file)
            return None
    
    def _process_hdf_with_gdal(self, hdf_file):
        """Process MAIAC HDF file using GDAL"""
        if not GDAL_AVAILABLE:
            return None
        
        try:
            if self.verbose:
                print("Processing HDF file with GDAL")
                
            gdal.UseExceptions()
            
            # Open HDF file
            hdf_ds = gdal.Open(hdf_file)
            if not hdf_ds:
                if self.verbose:
                    print("Failed to open HDF file with GDAL")
                return None
            
            # Get subdatasets
            subdatasets = hdf_ds.GetSubDatasets()
            
            if self.verbose:
                print(f"Found {len(subdatasets)} subdatasets")
                for i, subds in enumerate(subdatasets):
                    print(f"  {i}: {subds[0]} - {subds[1]}")
            
            # Find AOD subdataset
            aod_subds = None
            for subds in subdatasets:
                if 'Optical_Depth_055' in subds[1]:
                    aod_subds = subds[0]
                    if self.verbose:
                        print(f"Found AOD subdataset: {subds[1]}")
                    break
            
            # If not found, try other common names
            if aod_subds is None:
                for subds in subdatasets:
                    if any(term in subds[1] for term in ['AOD', 'Optical', 'Aerosol']):
                        aod_subds = subds[0]
                        if self.verbose:
                            print(f"Found alternative AOD subdataset: {subds[1]}")
                        break
            
            # If not found, use first subdataset as fallback
            if aod_subds is None and subdatasets:
                aod_subds = subdatasets[0][0]
                if self.verbose:
                    print(f"Using first subdataset as fallback: {subdatasets[0][1]}")
            
            if aod_subds:
                # Open subdataset
                ds = gdal.Open(aod_subds)
                if ds:
                    # Get metadata to find scale factor
                    metadata = ds.GetMetadata()
                    scale_factor = 0.001  # Default scale factor for MAIAC AOD
                    
                    if self.verbose:
                        print("Metadata keys:", list(metadata.keys()))
                        
                    # Look for scale factor in metadata
                    for key in metadata:
                        if 'scale' in key.lower():
                            try:
                                scale_factor = float(metadata[key])
                                if self.verbose:
                                    print(f"Found scale factor in metadata: {scale_factor}")
                            except:
                                if self.verbose:
                                    print(f"Could not convert scale factor '{metadata[key]}' to float")
                    
                    # Read data
                    band = ds.GetRasterBand(1)
                    img = band.ReadAsArray()
                    
                    # Find fill value or invalid value in metadata
                    fill_value = -9999  # Default fill value
                    for key in metadata:
                        if any(term in key.lower() for term in ['fill', 'nodata', '_fillvalue']):
                            try:
                                fill_value = float(metadata[key])
                                if self.verbose:
                                    print(f"Found fill value in metadata: {fill_value}")
                            except:
                                if self.verbose:
                                    print(f"Could not convert fill value '{metadata[key]}' to float")
                    
                    # Close datasets
                    band = None
                    ds = None
                    hdf_ds = None
                    
                    # Convert multi-dimensional to 2D if needed
                    if len(img.shape) > 2:
                        if self.verbose:
                            print(f"Converting multi-dimensional data {img.shape} to 2D")
                        img = img[:,:,0]  # Take first layer
                    
                    # Handle fill values - set to 0 as is standard for AOD missing data
                    img = img.astype(np.float32)
                    img[img == fill_value] = 0
                    
                    # Handle negative values - set to 0 as AOD cannot be negative
                    img[img < 0] = 0
                    
                    # Apply scale factor to get physical AOD values
                    # AOD values are typically between 0 and 5
                    if np.max(img) > 10:  # If values are still in raw count form
                        img = img * scale_factor
                        if self.verbose:
                            print(f"Applied scale factor {scale_factor} to AOD data")
                    
                    # Print statistics for debugging
                    if self.verbose:
                        print(f"AOD data statistics:")
                        print(f"  Min: {np.min(img)}")
                        print(f"  Max: {np.max(img)}")
                        print(f"  Mean: {np.mean(img)}")
                        zero_percent = np.sum(img == 0) / img.size * 100
                        print(f"  Zero pixels: {zero_percent:.1f}%")
                        unique_vals = np.unique(img)
                        print(f"  Unique values: {len(unique_vals)} " + 
                              (f"(first 10: {unique_vals[:10]})" if len(unique_vals) > 10 else f"({unique_vals})"))
                    
                    # Resize to target dimensions
                    aod_resized = cv2.resize(img, (self.dim, self.dim))
                    
                    return aod_resized
        
        except Exception as e:
            if self.verbose:
                print(f"Error in GDAL processing: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def _process_hdf_with_pyhdf(self, hdf_file):
        """Process MAIAC HDF file using pyhdf"""
        if not PYHDF_AVAILABLE:
            return None
        
        try:
            if self.verbose:
                print("Processing HDF file with pyhdf")
            
            # Open the HDF file
            hdf = SD(hdf_file, SDC.READ)
            
            # List all datasets
            datasets = hdf.datasets()
            
            if self.verbose:
                print(f"Found {len(datasets)} datasets:")
                for idx, (dsname, dsinfo) in enumerate(datasets.items()):
                    print(f"  {idx}: {dsname} - {dsinfo}")
            
            # Look for AOD dataset
            aod_ds_name = None
            for dsname in datasets.keys():
                if any(term in dsname for term in ['Optical_Depth_055', 'AOD', 'Optical', 'Aerosol']):
                    aod_ds_name = dsname
                    if self.verbose:
                        print(f"Found AOD dataset: {dsname}")
                    break
            
            # If not found, use first dataset as fallback
            if aod_ds_name is None and len(datasets) > 0:
                aod_ds_name = list(datasets.keys())[0]
                if self.verbose:
                    print(f"Using first dataset as fallback: {aod_ds_name}")
            
            if aod_ds_name:
                # Get the dataset
                aod_ds = hdf.select(aod_ds_name)
                
                # Get attributes to find scale factor and fill value
                attrs = aod_ds.attributes()
                
                if self.verbose:
                    print("Dataset attributes:", attrs)
                
                # Look for scale factor
                scale_factor = 0.001  # Default scale factor for MAIAC AOD
                for attr_name, attr_value in attrs.items():
                    if 'scale' in attr_name.lower():
                        try:
                            scale_factor = float(attr_value)
                            if self.verbose:
                                print(f"Found scale factor in attributes: {scale_factor}")
                        except:
                            if self.verbose:
                                print(f"Could not convert scale factor '{attr_value}' to float")
                
                # Look for fill value
                fill_value = -9999  # Default fill value
                for attr_name, attr_value in attrs.items():
                    if any(term in attr_name.lower() for term in ['fill', 'nodata', '_fillvalue']):
                        try:
                            fill_value = float(attr_value)
                            if self.verbose:
                                print(f"Found fill value in attributes: {fill_value}")
                        except:
                            if self.verbose:
                                print(f"Could not convert fill value '{attr_value}' to float")
                
                # Read the data
                img = aod_ds.get()
                
                # Close the file
                hdf.end()
                
                # Handle multi-dimensional data
                if len(img.shape) > 2:
                    if self.verbose:
                        print(f"Converting multi-dimensional data {img.shape} to 2D")
                    img = img[:,:,0]  # Take first layer
                
                # Handle fill values - set to 0 as is standard for AOD missing data
                img = img.astype(np.float32)
                img[img == fill_value] = 0
                
                # Handle negative values - set to 0 as AOD cannot be negative
                img[img < 0] = 0
                
                # Apply scale factor to get physical AOD values
                if np.max(img) > 10:  # If values are still in raw count form
                    img = img * scale_factor
                    if self.verbose:
                        print(f"Applied scale factor {scale_factor} to AOD data")
                
                # Print statistics for debugging
                if self.verbose:
                    print(f"AOD data statistics:")
                    print(f"  Min: {np.min(img)}")
                    print(f"  Max: {np.max(img)}")
                    print(f"  Mean: {np.mean(img)}")
                    zero_percent = np.sum(img == 0) / img.size * 100
                    print(f"  Zero pixels: {zero_percent:.1f}%")
                    unique_vals = np.unique(img)
                    print(f"  Unique values: {len(unique_vals)} " + 
                          (f"(first 10: {unique_vals[:10]})" if len(unique_vals) > 10 else f"({unique_vals})"))
                
                # Resize to target dimensions
                aod_resized = cv2.resize(img, (self.dim, self.dim))
                
                return aod_resized
        
        except Exception as e:
            if self.verbose:
                print(f"Error in pyhdf processing: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def _process_hdf_with_skimage(self, hdf_file):
        """Process MAIAC HDF file using skimage - matches approach in pm25Script_noGCN.py"""
        if not SKIMAGE_AVAILABLE:
            return None
        
        try:
            if self.verbose:
                print("Processing HDF file with skimage (same as pm25Script_noGCN.py)")
            
            # This is the exact approach used in pm25Script_noGCN.py
            img = io.imread(hdf_file)
            
            if self.verbose:
                print(f"Image shape after reading: {img.shape}")
                print(f"Image dtype: {img.dtype}")
            
            if len(img.shape) > 2:
                # Use rgb2gray if image has multiple channels
                img = rgb2gray(img)
                if self.verbose:
                    print(f"Converted to grayscale: {img.shape}")
            
            # Apply scale factor (0.001 is standard for MAIAC AOD)
            scale_factor = 0.001
            img = img.astype(np.float32) * scale_factor
            
            # Handle negative values - set to 0 as AOD cannot be negative
            img[img < 0] = 0
            
            if self.verbose:
                print(f"Applied scale factor {scale_factor}. New range: {np.min(img)} to {np.max(img)}")
            
            # Resize to the desired dimensions
            maiac = cv2.resize(img, (self.dim, self.dim))
            
            return maiac
            
        except Exception as e:
            if self.verbose:
                print(f"Error in skimage processing: {e}")
                import traceback
                traceback.print_exc()
            return None
    
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
        """Return information about the channels in the dataset"""
        channel_order = [
            'Landsat_SR_B1',       # 0
            'Landsat_SR_B7',       # 1
            'Landsat_QA_AEROSOL',  # 2
            'MAIAC_AOD',           # 3
        ]
        
        return {
            'landsat_bands': ['SR_B1', 'SR_B7', 'SR_QA_AEROSOL'],
            'maiac_bands': ['AOD_550nm'],
            'total_channels': len(channel_order),
            'channel_order': channel_order
        }
    
    def visualize_combined_data(self, data, channel_names):
        """Visualize all channels in the combined data with physical value ranges."""
        if not self.verbose:
            return
            
        num_channels = data.shape[-1]
        
        # Calculate subplot grid dimensions
        cols = min(3, num_channels)
        rows = (num_channels + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
        
        # Convert to 1D array for easier indexing
        axes = np.array(axes).ravel()
        
        for i, channel_name in enumerate(channel_names):
            if i < num_channels:
                ax = axes[i]
                channel_data = data[0, 0, :, :, i]
                
                # Calculate appropriate min/max based on physical meaning of the data
                if "MAIAC" in channel_name or "AOD" in channel_name:
                    # AOD values typically range from 0 to 5
                    vmin = 0
                    vmax = min(5.0, np.max(channel_data)) if np.max(channel_data) > 0 else 1.0
                else:
                    # For Landsat data, use data-driven approach
                    non_zero = channel_data[channel_data != 0]
                    if len(non_zero) > 0:
                        vmin = np.min(non_zero)
                        vmax = np.percentile(non_zero, 99)
                    else:
                        vmin = 0
                        vmax = 1

                normalized_data = (channel_data - vmin) / (vmax - vmin)
                im = ax.imshow(normalized_data, cmap='viridis')
                ax.set_title(f"Channel {i}: {channel_name}")
                plt.colorbar(im, ax=ax)
                ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_channels, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'combined_channels_visualization.png')
        plt.savefig(output_file)
        plt.close()
        
        print(f"Visualization saved to '{output_file}'")
    
    def _print_data_statistics(self):
        """Print detailed statistics about the final data"""
        if not self.verbose:
            return
        
        print("\nData Statistics:")
        print("===============")
        
        print("\nMAIAC Data:")
        maiac_data = self.maiac_data
        
        # Get first 10 timestamps
        days_to_check = min(10, maiac_data.shape[0])
        
        for day_idx in range(days_to_check):
            day_data = maiac_data[day_idx, :, :, 0]
            non_zero = np.count_nonzero(day_data)
            if non_zero > 0:
                print(f"  Day {day_idx} stats:")
                print(f"    Min: {np.min(day_data):.5f}")
                print(f"    Max: {np.max(day_data):.5f}")
                print(f"    Mean: {np.mean(day_data):.5f}")
                print(f"    Data coverage: {non_zero/day_data.size*100:.2f}%")
        
        print("\nLandsat Data:")
        landsat_data = self.landsat_data
        
        # Calculate Landsat statistics for each band
        for band_idx in range(landsat_data.shape[3]):
            band_data = landsat_data[0, :, :, band_idx]
            non_zero = np.count_nonzero(band_data)
            print(f"  Band {band_idx} stats:")
            print(f"    Min: {np.min(band_data)}")
            print(f"    Max: {np.max(band_data)}")
            print(f"    Mean: {np.mean(band_data)}")
            print(f"    Data coverage: {non_zero/band_data.size*100:.2f}%")
        
        print("\nFinal Data Shape:")
        print(f"  {self.data.shape[0]} samples")
        print(f"  {self.data.shape[1]} frames per sample")
        print(f"  {self.data.shape[2]}x{self.data.shape[3]} grid size")
        print(f"  {self.data.shape[4]} channels")

    def save_data(self, filepath):
        """Save the processed data to a file"""
        np.save(filepath, self.data)
        if self.verbose:
            print(f"Data saved to {filepath}")
    
    def load_data(self, filepath):
        """Load processed data from a file"""
        self.data = np.load(filepath)
        if self.verbose:
            print(f"Data loaded from {filepath}")
            print(f"Data shape: {self.data.shape}")