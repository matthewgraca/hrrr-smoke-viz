# test_combined_data.py
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
from datetime import datetime
from dotenv import load_dotenv  # Add this import

# Add path to libs directory
sys.path.append("../libs")  # Adjust path as needed

from airnowdata import AirNowData
from hrrrdata import HRRRData
from pwwbdata import PWWBData

def test_combined_pipeline():
    """Test running all three data sources and combining them"""
    load_dotenv()
    # Configuration
    config = {
        'start_date': "2024-12-01",
        'end_date': "2024-12-31",
        'extent': (-118.4, -118.0, 33.9, 34.2),  # LA area
        'frames_per_sample': 5,
        'dim': 200,
        'env_file': '.env'
    }
    print("=== Testing Combined Data Pipeline ===")
    print(f"Date range: {config['start_date']} to {config['end_date']}")
    print(f"Spatial extent: {config['extent']}")
    print(f"Grid dimensions: {config['dim']}x{config['dim']}")
    print()
    
    # 1. Test AirNow Data
    print("1. Loading AirNow data...")
    airnow = AirNowData(
        start_date=config['start_date'],
        end_date=config['end_date'],
        extent=config['extent'],
        airnow_api_key=os.getenv('AIRNOW_API_KEY'),
        frames_per_sample=config['frames_per_sample'],
        dim=config['dim'],
        create_elevation_mask=False,  # Create sample elevation and mask data
        force_reprocess=False  # Use cached data if available
    )
    airnow_data = airnow.data
    air_sens_loc = airnow.air_sens_loc
    target_stations = airnow.target_stations
    
    print(f"✓ AirNow data shape: {airnow_data.shape}")
    print(f"  - Included {len(air_sens_loc)} sensor locations")
    if target_stations is not None:
        print(f"  - Target stations shape: {target_stations.shape}")
    else:
        print("  - No target stations available")
    
    # 2. Test HRRR Data
    print("\n2. Loading HRRR data...")
    try:
        hrrr = HRRRData(
            start_date=config['start_date'],
            end_date=config['end_date'],
            extent=config['extent'],
            extent_name='la_region',
            product='MASSDEN',
            frames_per_sample=config['frames_per_sample'],
            dim=config['dim'],
            verbose=True
        )
        # Convert units
        ug_per_kg = 1000000000
        hrrr_data = hrrr.data * ug_per_kg
        print(f"✓ HRRR data shape: {hrrr_data.shape}")
        print(f"  - HRRR stats: min={hrrr_data.min():.6f}, max={hrrr_data.max():.6f}, mean={hrrr_data.mean():.6f}")
    except Exception as e:
        print(f"✗ HRRR failed: {e}")
        return
    
    # 3. Test PWWB Data
    print("\n3. Loading PWWB channels...")
    try:
        pwwb = PWWBData(
            start_date=config['start_date'],
            end_date=config['end_date'],
            extent=config['extent'],
            frames_per_sample=config['frames_per_sample'],
            dim=config['dim'],
            env_file=config['env_file'],
            verbose=True,
            include_wind=False  # Skip wind data
        )
        print(f"✓ PWWB data shape: {pwwb.data.shape}")
        channel_info = pwwb.get_channel_info()
        print(f"  - Channels: {channel_info['channel_order']}")
        
        # Add debug visualization of raw PWWB data
        print("\nDebug visualization of raw PWWB data:")
        sample_idx = 0
        frame_idx = 0
        fig, axes = plt.subplots(1, pwwb.data.shape[-1], figsize=(5*pwwb.data.shape[-1], 5))
        
        # Handle case of a single channel
        if pwwb.data.shape[-1] == 1:
            axes = [axes]
            
        for c in range(pwwb.data.shape[-1]):
            channel_data = pwwb.data[sample_idx, frame_idx, :, :, c]
            vmin = channel_data.min()
            vmax = channel_data.max()
            
            print(f"  Channel {c} ({channel_info['channel_order'][c]}): min={vmin:.3f}, max={vmax:.3f}, mean={channel_data.mean():.3f}")
            
            ax = axes[c]
            im = ax.imshow(channel_data, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(channel_info['channel_order'][c])
            ax.axis('off')
            plt.colorbar(im, ax=ax)
        
        plt.suptitle("Raw PWWB Data (Before Combination)")
        plt.tight_layout()
        plt.savefig("pwwb_raw_data.png")
        print("Raw PWWB data visualization saved to pwwb_raw_data.png")
    except Exception as e:
        print(f"✗ PWWB failed: {e}")
        return
    
    # 4. Combine all channels to match original PWWB model
    print("\n4. Combining all channels...")
    try:
        # Split PWWB channels for Landsat and MAIAC
        landsat_maiac = pwwb.data  # All PWWB channels
        
        # Create wind data placeholder (only if not using real wind data)
        wind_data = np.zeros_like(airnow_data)
        print(f"  - Using wind data placeholder with shape {wind_data.shape}")
        
        # Combine in the correct order
        combined_data = np.concatenate([
            landsat_maiac,   # Channels 0-4: Landsat (3) + MAIAC (1)
            airnow_data,     # Channel 5: AirNow PM2.5 
            wind_data,       # Channel 6: Wind Speed (placeholder)
            hrrr_data        # Channel 7: HRRR (additional)
        ], axis=-1)
        
        print(f"✓ Combined data shape: {combined_data.shape}")
        print(f"  - Total channels: {combined_data.shape[-1]}")
        
        # Determine channel map based on what's actually in the data
        channel_map = {}
        
        # PWWB channels (first 4) - always include Landsat and MAIAC
        for i, name in enumerate(channel_info['channel_order']):
            channel_map[i] = name
        
        # Add AirNow, Wind and HRRR
        channel_map[len(channel_map)] = "AirNow PM2.5"
        channel_map[len(channel_map)] = "Wind Speed (Placeholder)"
        channel_map[len(channel_map)] = "HRRR Mass Density"
        
        print("\nChannel mapping:")
        for idx, name in channel_map.items():
            print(f"  Channel {idx}: {name}")
    except Exception as e:
        print(f"✗ Combination failed: {e}")
        return
    
    # 5. Visualize combined data
    print("\n5. Visualizing combined data...")
    try:
        num_channels = combined_data.shape[-1]
        rows = (num_channels + 3) // 4  # 4 columns max
        cols = min(4, num_channels)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
        axes = axes.ravel()  # Flatten array for easier indexing
        
        sample_idx = 0
        frame_idx = 0
        
        for i in range(num_channels):
            ax = axes[i]
            channel_data = combined_data[sample_idx, frame_idx, :, :, i]
            
            # Calculate appropriate min/max for each channel
            if "MAIAC" in channel_map[i]:
                # Special handling for MAIAC channel
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
            
            # Print channel stats for debugging
            print(f"  Channel {i} ({channel_map[i]}): min={channel_data.min():.3f}, max={channel_data.max():.3f}, mean={channel_data.mean():.3f}")
            print(f"    Using display range: vmin={vmin:.3f}, vmax={vmax:.3f}")
            
            im = ax.imshow(channel_data, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(channel_map[i])
            ax.axis('off')
            plt.colorbar(im, ax=ax)
        
        # Hide unused subplots
        for i in range(num_channels, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Combined Data Channels - Sample {sample_idx}, Frame {frame_idx}')
        plt.tight_layout()
        plt.savefig('combined_channels_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Visualization saved to 'combined_channels_visualization.png'")
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
    
    # 6. Verify data integrity
    print("\n6. Verifying data integrity...")
    try:
        # Check for NaN values
        nan_count = np.isnan(combined_data).sum()
        print(f"  - NaN values: {nan_count}")
        
        # Check value ranges for each channel
        print("  - Value ranges per channel:")
        for i in range(combined_data.shape[-1]):
            channel_data = combined_data[:, :, :, :, i]
            print(f"    Channel {i} ({channel_map[i]}): "
                  f"min={channel_data.min():.3f}, max={channel_data.max():.3f}, "
                  f"mean={channel_data.mean():.3f}, std={channel_data.std():.3f}")
        
        # Check shapes consistency
        sample_count = combined_data.shape[0]
        print(f"✓ Sample count: {sample_count}")
        
        print("\n✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")
    
    return combined_data
if __name__ == "__main__":
    # Check for required files
    print("Checking for required files...")
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        print("Creating sample .env file...")
        with open('.env', 'w') as f:
            f.write("""# USGS M2M API credentials
M2M_USERNAME=your_username
M2M_TOKEN=your_token

# NASA MAIAC credentials
MAIAC_USERNAME=your_username
MAIAC_PASSWORD=your_password

# AirNow API key
AIRNOW_API_KEY=your_api_key

# OpenWeatherMap API key
OPENWEATHER_API_KEY=your_api_key
""")
        print("Please update .env with your actual credentials")
        sys.exit(1)
    
    # Run the test
    combined_data = test_combined_pipeline()