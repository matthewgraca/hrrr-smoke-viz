import requests
import pandas as pd
import numpy as np
import cv2
import io
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt

class MetarWindData:
    def __init__(
        self,
        start_date="2023-08-02-00",
        end_date="2023-08-02-12",
        extent=(-118.75, -117.0, 33.5, 34.5),
        save_dir='data/metar_wind.csv',
        processed_cache_dir='data/metar_wind_processed.npz',
        dim=40,
        idw_power=2.0,
        use_elevation=True,
        elevation_path='inputs/elevation.npy',
        min_uptime=0.25,
        zscore_threshold=3,
        force_reprocess=False,
        visualize_samples=3,
        verbose=1,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.extent = extent
        self.dim = dim
        self.idw_power = idw_power
        self.use_elevation = use_elevation
        self.min_uptime = min_uptime
        self.zscore_threshold = zscore_threshold
        self.verbose = verbose
        self.visualize_samples = visualize_samples
        
        if use_elevation:
            if os.path.exists(elevation_path):
                elevation_raw = np.load(elevation_path)
                if elevation_raw.shape != (dim, dim):
                    elevation_raw = cv2.resize(elevation_raw, (dim, dim))
                self.elevation = self._normalize_elevation(elevation_raw)
                if verbose:
                    print(f"Using 3D IDW with elevation from {elevation_path}")
            else:
                raise FileNotFoundError(f"use_elevation=True but elevation data not found at {elevation_path}")
        else:
            self.elevation = np.zeros((dim, dim), dtype=np.float32)
            if verbose:
                print("Using 2D IDW (no elevation)")
        
        if not force_reprocess and os.path.exists(processed_cache_dir):
            if verbose:
                print(f"Loading from cache: {processed_cache_dir}")
            try:
                cached = np.load(processed_cache_dir, allow_pickle=True)
                self.data = cached['data']
                self.ground_site_grids = cached['ground_site_grids']
                self.sensor_locations = cached['sensor_locations'].item()
                self.timestamps = cached['timestamps']
                if verbose:
                    print(f" Loaded data shape: {self.data.shape}")
                return
            except Exception as e:
                print(f"Cache load failed: {e}. Reprocessing...")
        
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        os.makedirs(os.path.dirname(processed_cache_dir), exist_ok=True)
        
        df = self._fetch_metar_wind_data(start_date, end_date, extent, save_dir)
        
        if df.empty:
            raise ValueError("No METAR data retrieved")
        
        df = self._filter_low_uptime_stations(df, start_date, end_date)
        df = self._convert_wind_to_ms(df)
        df = self._remove_outliers(df, zscore_threshold)
        df = self._compute_wind_components(df)
        df = self._forward_fill_missing(df)
        
        self.sensor_locations = {}
        lon_min, lon_max, lat_min, lat_max = extent
        lat_dist = abs(lat_max - lat_min)
        lon_dist = abs(lon_max - lon_min)
        
        for station_id in df['station'].unique():
            station_data = df[df['station'] == station_id].iloc[0]
            
            if pd.isna(station_data['lat']) or pd.isna(station_data['lon']):
                if verbose:
                    print(f"  Skipping {station_id}: missing coordinates")
                continue
            
            x = int(((lat_max - station_data['lat']) / lat_dist) * dim)
            y = int(((station_data['lon'] - lon_min) / lon_dist) * dim)
            x = max(0, min(x, dim - 1))
            y = max(0, min(y, dim - 1))
            self.sensor_locations[station_id] = (x, y)
        
        if verbose:
            print(f"Active sensors: {len(self.sensor_locations)}")
        
        df['hour'] = df['valid'].dt.floor('h')
        
        self.timestamps = sorted(df['hour'].unique())
        
        station_coverage = df.groupby('station')['hour'].nunique() / len(self.timestamps)
        reliable_stations = station_coverage[station_coverage >= self.min_uptime].index
        df = df[df['station'].isin(reliable_stations)]
        
        if verbose:
            print(f"Kept {len(reliable_stations)} reliable stations (>={self.min_uptime*100}% coverage)")
        
        all_hours = pd.DataFrame({'hour': self.timestamps})
        all_stations = pd.DataFrame({'station': reliable_stations})
        complete_grid = all_hours.assign(key=1).merge(all_stations.assign(key=1), on='key').drop('key', axis=1)
        
        hourly_data = df.groupby(['station', 'hour']).agg({
            'sped': 'mean',
            'u_wind': 'mean',
            'v_wind': 'mean',
            'lat': 'first',
            'lon': 'first'
        }).reset_index()
        
        full_data = complete_grid.merge(hourly_data, on=['station', 'hour'], how='left')
        
        full_data = full_data.sort_values(['station', 'hour'])
        full_data[['sped', 'u_wind', 'v_wind']] = full_data.groupby('station')[['sped', 'u_wind', 'v_wind']].ffill()
        
        full_data[['sped', 'u_wind', 'v_wind']] = full_data.groupby('station')[['sped', 'u_wind', 'v_wind']].bfill()
        
        full_data['lat'] = full_data.groupby('station')['lat'].transform('first')
        full_data['lon'] = full_data.groupby('station')['lon'].transform('first')
        
        all_grids = []
        
        if verbose:
            print(f"Creating grids for {len(self.timestamps)} hours with {len(reliable_stations)} stations each")
        
        for timestamp in (tqdm(self.timestamps, desc="Gridding") if verbose else self.timestamps):
            hour_data = full_data[full_data['hour'] == timestamp]
            
            speed_grid = self._create_station_grid_from_df(hour_data, 'sped')
            u_grid = self._create_station_grid_from_df(hour_data, 'u_wind')
            v_grid = self._create_station_grid_from_df(hour_data, 'v_wind')
            
            all_grids.append(np.stack([speed_grid, u_grid, v_grid], axis=-1))
        
        self.ground_site_grids = np.array(all_grids)
        
        if verbose:
            print("Applying IDW interpolation...")
        
        interpolated = []
        for frame in (tqdm(self.ground_site_grids, desc="Interpolating") if verbose else self.ground_site_grids):
            interp_frame = np.zeros_like(frame)
            for var_idx in range(3):
                interp_frame[:, :, var_idx] = self._idw_interpolate_grid(frame[:, :, var_idx])
            interpolated.append(interp_frame)
        
        self.data = np.array(interpolated)
        
        if visualize_samples > 0:
            self._visualize_random_samples(visualize_samples)
        
        if verbose:
            print(f"Saving to cache: {processed_cache_dir}")
        np.savez_compressed(
            processed_cache_dir,
            data=self.data,
            ground_site_grids=self.ground_site_grids,
            sensor_locations=np.array([self.sensor_locations]),
            timestamps=np.array(self.timestamps),
        )
    
    def _normalize_elevation(self, elevation_data):
        """Normalize elevation to 0-100 range."""
        min_val = np.min(elevation_data)
        max_val = np.max(elevation_data)
        
        if max_val == min_val:
            return np.zeros_like(elevation_data)
        
        normalized = (elevation_data - min_val) / (max_val - min_val) * 100
        return normalized.astype(np.float32)
    
    def _fetch_metar_wind_data(self, start_date, end_date, extent, save_dir):
        """Fetch wind data from IEM METAR network."""
        
        if os.path.exists(save_dir):
            if self.verbose:
                print(f"Loading existing data: {save_dir}")
            df = pd.read_csv(save_dir)
            df['valid'] = pd.to_datetime(df['valid'])
            
            if df['lat'].isna().all() or df['lon'].isna().all():
                if self.verbose:
                    print("Cached CSV missing coordinates, re-fetching...")
                os.remove(save_dir)
                return self._fetch_metar_wind_data(start_date, end_date, extent, save_dir)
            
            return df
        
        lon_min, lon_max, lat_min, lat_max = extent
        
        if self.verbose:
            print("Querying METAR network for stations...")
        
        stations_url = "https://mesonet.agron.iastate.edu/geojson/network/CA_ASOS.geojson"
        response = requests.get(stations_url, timeout=30)
        geojson = response.json()
        
        stations = []
        for feature in geojson['features']:
            props = feature['properties']
            lon, lat = feature['geometry']['coordinates']
            
            if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                stations.append({
                    'id': props['sid'],
                    'name': props.get('sname', 'Unknown'),
                    'lat': lat,
                    'lon': lon
                })
        
        if self.verbose:
            print(f"Found {len(stations)} stations in extent")
        
        station_ids = [s['id'] for s in stations]
        
        start_dt = pd.to_datetime(start_date, format="%Y-%m-%d-%H")
        end_dt = pd.to_datetime(end_date, format="%Y-%m-%d-%H")
        
        url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
        params = {
            'station': ','.join(station_ids),
            'data': ['sped', 'drct'],
            'year1': start_dt.year,
            'month1': start_dt.month,
            'day1': start_dt.day,
            'hour1': start_dt.hour,
            'year2': end_dt.year,
            'month2': end_dt.month,
            'day2': end_dt.day,
            'hour2': end_dt.hour,
            'tz': 'Etc/UTC',
            'format': 'comma',
            'latlon': 'no',
            'report_type': '3'
        }
        
        if self.verbose:
            print(f"Fetching wind data from {start_dt} to {end_dt}...")
        
        response = requests.post(url, data=params, timeout=300)
        df = pd.read_csv(io.StringIO(response.text), comment='#')
        df['valid'] = pd.to_datetime(df['valid'])
        
        df['sped'] = pd.to_numeric(df['sped'], errors='coerce')
        df['drct'] = pd.to_numeric(df['drct'], errors='coerce')
        
        station_lookup = {}
        for s in stations:
            station_lookup[s['id']] = s
            station_lookup[s['id'].replace('K', '')] = s
            if not s['id'].startswith('K'):
                station_lookup['K' + s['id']] = s
        
        df['lat'] = df['station'].map(lambda x: station_lookup.get(x, {}).get('lat'))
        df['lon'] = df['station'].map(lambda x: station_lookup.get(x, {}).get('lon'))
        df['name'] = df['station'].map(lambda x: station_lookup.get(x, {}).get('name'))
        
        df = df.dropna(subset=['lat', 'lon'])
        
        df.to_csv(save_dir, index=False)
        if self.verbose:
            print(f" Saved to {save_dir}")
        
        return df
    
    def _filter_low_uptime_stations(self, df, start_date, end_date):
        """Remove stations with uptime < min_uptime threshold."""
        start_dt = pd.to_datetime(start_date, format="%Y-%m-%d-%H")
        end_dt = pd.to_datetime(end_date, format="%Y-%m-%d-%H")
        
        expected_readings = len(pd.date_range(start_dt, end_dt, freq='h', inclusive='left'))
        
        station_counts = df.groupby('station').size()
        valid_stations = station_counts[station_counts / expected_readings >= self.min_uptime].index
        
        filtered = df[df['station'].isin(valid_stations)].copy()
        
        if self.verbose:
            removed = len(df['station'].unique()) - len(valid_stations)
            print(f"Filtered out {removed} low-uptime stations (< {self.min_uptime*100}%)")
        
        return filtered
    
    def _convert_wind_to_ms(self, df):
        """Convert wind speed from knots to m/s."""
        df['sped'] = df['sped'] * 0.514444
        return df
    
    def _remove_outliers(self, df, zscore_threshold=3):
        """Remove outliers using z-score per station."""
        df_clean = df.copy()
        
        wind_vars = ['sped']
        for var in wind_vars:
            if var in df_clean.columns:
                station_groups = df_clean.groupby('station')[var]
                
                means = station_groups.transform('mean')
                stds = station_groups.transform('std')
                stds = stds.replace(0, 1)
                
                z_scores = (df_clean[var] - means) / stds
                
                df_clean.loc[np.abs(z_scores) > zscore_threshold, var] = np.nan
        
        if self.verbose:
            total_before = df[wind_vars].notna().sum().sum()
            total_after = df_clean[wind_vars].notna().sum().sum()
            total_outliers = total_before - total_after
            print(f"Removed {total_outliers} outliers (z-score > {zscore_threshold})")
        
        return df_clean
    
    def _compute_wind_components(self, df):
        """Compute U and V wind components from speed and direction."""
        wind_dir_rad = np.deg2rad(df['drct'])
        df['u_wind'] = -df['sped'] * np.sin(wind_dir_rad)
        df['v_wind'] = -df['sped'] * np.cos(wind_dir_rad)
        
        df.loc[df['sped'].isna() | df['drct'].isna(), 'u_wind'] = np.nan
        df.loc[df['sped'].isna() | df['drct'].isna(), 'v_wind'] = np.nan
        
        return df
    
    def _forward_fill_missing(self, df):
        """Forward fill missing wind data for each station."""
        df_filled = df.sort_values(['station', 'valid']).copy()
        
        wind_vars = ['sped', 'drct', 'u_wind', 'v_wind']
        for var in wind_vars:
            if var in df_filled.columns:
                df_filled[var] = df_filled.groupby('station')[var].ffill()
        
        return df_filled
    
    def _create_station_grid(self, ts_data, variable):
        """Place station measurements on grid."""
        grid = np.full((self.dim, self.dim), np.nan)
        
        lon_min, lon_max, lat_min, lat_max = self.extent
        lat_dist = abs(lat_max - lat_min)
        lon_dist = abs(lon_max - lon_min)
        
        for _, row in ts_data.iterrows():
            if pd.isna(row[variable]):
                continue
            
            x = int(((lat_max - row['lat']) / lat_dist) * self.dim)
            y = int(((row['lon'] - lon_min) / lon_dist) * self.dim)
            x = max(0, min(x, self.dim - 1))
            y = max(0, min(y, self.dim - 1))
            
            grid[x, y] = row[variable]
        
        return grid
    
    def _create_station_grid_from_df(self, df, variable):
        """Place station measurements on grid from a DataFrame."""
        grid = np.full((self.dim, self.dim), np.nan)
        
        lon_min, lon_max, lat_min, lat_max = self.extent
        lat_dist = abs(lat_max - lat_min)
        lon_dist = abs(lon_max - lon_min)
        
        for _, row in df.iterrows():
            if pd.isna(row[variable]):
                continue
            
            x = int(((lat_max - row['lat']) / lat_dist) * self.dim)
            y = int(((row['lon'] - lon_min) / lon_dist) * self.dim)
            x = max(0, min(x, self.dim - 1))
            y = max(0, min(y, self.dim - 1))
            
            grid[x, y] = row[variable]
        
        return grid
    
    def _idw_interpolate_grid(self, grid, k_neighbors=10):
        sensor_coords = [(x, y) for x, y in zip(*np.where(~np.isnan(grid)))]
        
        if not sensor_coords:
            return np.zeros_like(grid)
        
        if len(sensor_coords) == 1:
            return np.full_like(grid, grid[sensor_coords[0]])
        
        interpolated = grid.copy()
        
        for x in range(self.dim):
            for y in range(self.dim):
                if not np.isnan(grid[x, y]):
                    continue
                
                distances = []
                values = []
                
                for sx, sy in sensor_coords:
                    if self.use_elevation:
                        dx = x - sx
                        dy = y - sy
                        dz = (self.elevation[x, y] - self.elevation[sx, sy]) * 2
                        dist = np.sqrt(dx**2 + dy**2 + dz**2)
                    else:
                        dist = np.sqrt((x - sx)**2 + (y - sy)**2)
                    
                    if dist == 0:
                        dist = 1e-10
                        
                    distances.append(dist)
                    values.append(grid[sx, sy])
                
                distances = np.array(distances)
                values = np.array(values)
                
                k = min(k_neighbors, len(distances))
                nearest_indices = np.argpartition(distances, k-1)[:k]
                nearest_distances = distances[nearest_indices]
                nearest_values = values[nearest_indices]
                
                weights = 1 / nearest_distances**self.idw_power
                weights /= np.sum(weights)
                
                interpolated[x, y] = np.sum(weights * nearest_values)
        
        return interpolated
    
    def _visualize_random_samples(self, n_samples):
        if len(self.data) == 0:
            return
        
        indices = np.random.choice(len(self.data), min(n_samples, len(self.data)), replace=False)
        
        for idx in indices:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            speed_raw = self.ground_site_grids[idx, :, :, 0]
            u_raw = self.ground_site_grids[idx, :, :, 1]
            v_raw = self.ground_site_grids[idx, :, :, 2]
            
            speed_raw_masked = np.ma.masked_invalid(speed_raw)
            u_raw_masked = np.ma.masked_invalid(u_raw)
            v_raw_masked = np.ma.masked_invalid(v_raw)
            
            im1 = axes[0, 0].imshow(speed_raw_masked, cmap='viridis', vmin=0, vmax=10)
            axes[0, 0].set_title(f'Raw Speed (t={idx})')
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
            
            im2 = axes[0, 1].imshow(u_raw_masked, cmap='RdBu_r', vmin=-10, vmax=10)
            axes[0, 1].set_title('Raw U-wind')
            axes[0, 1].axis('off')
            plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
            
            im3 = axes[0, 2].imshow(v_raw_masked, cmap='RdBu_r', vmin=-10, vmax=10)
            axes[0, 2].set_title('Raw V-wind')
            axes[0, 2].axis('off')
            plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
            
            im4 = axes[1, 0].imshow(self.data[idx, :, :, 0], cmap='viridis', vmin=0, vmax=10)
            axes[1, 0].set_title('IDW Speed')
            axes[1, 0].axis('off')
            plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            im5 = axes[1, 1].imshow(self.data[idx, :, :, 1], cmap='RdBu_r', vmin=-10, vmax=10)
            axes[1, 1].set_title('IDW U-wind')
            axes[1, 1].axis('off')
            plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
            
            im6 = axes[1, 2].imshow(self.data[idx, :, :, 2], cmap='RdBu_r', vmin=-10, vmax=10)
            axes[1, 2].set_title('IDW V-wind')
            axes[1, 2].axis('off')
            plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)
            
            for x, y in self.sensor_locations.values():
                for ax in axes.flat:
                    ax.plot(y, x, 'r.', markersize=8)
            
            plt.tight_layout()
            plt.savefig(f'metar_wind_sample_{idx}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            if self.verbose:
                print(f" Saved visualization: metar_wind_sample_{idx}.png")


if __name__ == "__main__":
    metar = MetarWindData(
        start_date="2023-08-02-00",
        end_date="2023-08-02-12",
        extent=(-118.75, -117.0, 33.5, 34.5),
        dim=40,
        idw_power=2.0,
        elevation_path='inputs/elevation.npy',
        min_uptime=0.75,
        zscore_threshold=3,
        force_reprocess=True,
        visualize_samples=3,
        verbose=1
    )
    
    print(f"\nFinal data shape: {metar.data.shape}")
    print(f"Variables: [speed(m/s), u_wind(m/s), v_wind(m/s)]")
    print(f"Active sensors: {len(metar.sensor_locations)}")