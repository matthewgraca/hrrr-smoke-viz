import numpy as np
import rasterio
from rasterio.windows import from_bounds
from scipy import ndimage
import matplotlib.pyplot as plt

def load_data_convert_grid(tiff_file='la_elevation_map.tif', 
                                    lat_bottom=33.5, lat_top=34.5,
                                    lon_bottom=-118.75, lon_top=-117.0,
                                    target_size=(40, 40)): 
    print(f"Loading and processing: {tiff_file}")
    print(f"Target output shape: {target_size}")
    
    with rasterio.open(tiff_file) as src:
        window = from_bounds(lon_bottom, lat_bottom, lon_top, lat_top, src.transform)
        
        elevation_data = src.read(1, window=window)
        nodata = src.nodata
        
        print(f"Original subset shape: {elevation_data.shape}")
        print(f"NoData value: {nodata}")
    
    if nodata is not None:
        nodata_mask = (elevation_data == nodata) | (elevation_data < -100)
        elevation_data[nodata_mask] = 0
    else:
        nodata_mask = elevation_data < -100
        elevation_data[nodata_mask] = 0
    
    zoom_factors = (
        target_size[0] / elevation_data.shape[0],
        target_size[1] / elevation_data.shape[1]
    )
    
    print(f"Zoom factors: {zoom_factors}")
    
    resampled_data = ndimage.zoom(elevation_data, zoom_factors, order=1)
    
    resampled_data = resampled_data[:target_size[0], :target_size[1]]
    
    if resampled_data.shape != target_size:
        pad_height = target_size[0] - resampled_data.shape[0]
        pad_width = target_size[1] - resampled_data.shape[1]
        resampled_data = np.pad(resampled_data, 
                                ((0, pad_height), (0, pad_width)), 
                                mode='edge')
    
    metadata = {
        'original_shape': elevation_data.shape,
        'resampled_shape': resampled_data.shape,
        'bounds': {
            'lat_bottom': lat_bottom,
            'lat_top': lat_top,
            'lon_bottom': lon_bottom,
            'lon_top': lon_top
        },
        'original_min': float(np.min(elevation_data)),
        'original_max': float(np.max(elevation_data)),
        'original_mean': float(np.mean(elevation_data)),
        'resampled_min': float(np.min(resampled_data)),
        'resampled_max': float(np.max(resampled_data)),
        'resampled_mean': float(np.mean(resampled_data)),
        'units': 'meters',
        'normalized': False
    }
    
    print(f"\nRaw elevation data (no normalization)")
    print(f"  Range: [{np.min(resampled_data):.2f}, {np.max(resampled_data):.2f}] meters")
    print(f"  Mean: {np.mean(resampled_data):.2f} meters")
    print(f"  Std: {np.std(resampled_data):.2f} meters")
    
    return resampled_data, metadata


def visualize_elevation(elevation_data, metadata):
    axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(elevation_data, cmap='terrain', aspect='auto')
    axes[0].set_title(f'Elevation Data (40x40)\nRange: [{metadata["resampled_min"]:.1f}, {metadata["resampled_max"]:.1f}] meters')
    axes[0].set_xlabel('Grid X')
    axes[0].set_ylabel('Grid Y')
    plt.colorbar(im1, ax=axes[0], label='Elevation (meters)')
    
    X, Y = np.meshgrid(range(40), range(40))
    contour = axes[1].contour(X, Y, elevation_data, levels=15, cmap='viridis')
    axes[1].clabel(contour, inline=True, fontsize=8)
    im2 = axes[1].imshow(elevation_data, cmap='terrain', aspect='auto', alpha=0.6)
    axes[1].set_title('Elevation Contours (40x40)')
    axes[1].set_xlabel('Grid X')
    axes[1].set_ylabel('Grid Y')
    plt.colorbar(im2, ax=axes[1], label='Elevation (meters)')
    
    plt.suptitle(f'Raw Elevation: {metadata["bounds"]["lat_bottom"]}°N to {metadata["bounds"]["lat_top"]}°N, '
                 f'{metadata["bounds"]["lon_bottom"]}°E to {metadata["bounds"]["lon_top"]}°E',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('elevation_40x40_raw.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    print("=" * 60)
    print(" " * 10 + "RAW ELEVATION TO 40x40 CONVERTER")
    print("=" * 60)
    
    tiff_file = 'la_elevation_map.tif'
    lat_bottom, lat_top = 33.5, 34.5
    lon_bottom, lon_top = -118.75, -117.0
    
    print("\n1. Resampling to 40x40 grid (keeping raw elevation values)...")
    elevation_40x40_raw, metadata = load_data_convert_grid(
        tiff_file=tiff_file,
        lat_bottom=lat_bottom,
        lat_top=lat_top,
        lon_bottom=lon_bottom,
        lon_top=lon_top,
        target_size=(40, 40)
    )
    
    print("\n2. Raw Data Statistics:")
    print(f"   Shape: {elevation_40x40_raw.shape}")
    print(f"   Data type: {elevation_40x40_raw.dtype}")
    print(f"   Min elevation: {np.min(elevation_40x40_raw):.2f} meters")
    print(f"   Max elevation: {np.max(elevation_40x40_raw):.2f} meters")
    print(f"   Mean elevation: {np.mean(elevation_40x40_raw):.2f} meters")
    print(f"   Std deviation: {np.std(elevation_40x40_raw):.2f} meters")
    
    ocean_cells = np.sum(elevation_40x40_raw <= 0)
    land_cells = np.sum(elevation_40x40_raw > 0)
    print(f"\n3. Grid composition:")
    print(f"   Ocean/NoData cells (≤0m): {ocean_cells} ({ocean_cells/16:.1%})")
    print(f"   Land cells (>0m): {land_cells} ({land_cells/16:.1%})")
    
    print("\n4. Saving raw elevation array...")
    np.save('elevation_40x40_raw.npy', elevation_40x40_raw)
    print(f"   ✓ Saved raw 40x40 array to: elevation_40x40_raw.npy")
    
    np.save('elevation_metadata_raw.npy', metadata)
    print(f"   ✓ Saved metadata to: elevation_metadata_raw.npy")
    
    print("\n5. Creating visualization...")
    visualize_elevation(elevation_40x40_raw, metadata)
    
    print("\n" + "=" * 60)
    print(" " * 20 + "SUCCESS!")
    print("=" * 60)
    print("\nOutput files:")
    print("  - elevation_40x40_raw.npy: 40x40 raw elevation grid (meters)")
    print("  - elevation_metadata_raw.npy: Metadata with bounds and statistics")
    print("  - elevation_40x40_raw.png: Visualization")
    
    print("\nTo load the raw elevation data:")
    print("   elevation = np.load('elevation_40x40_raw.npy')")
    print("   metadata = np.load('elevation_metadata_raw.npy', allow_pickle=True).item()")
    print("\n   # elevation is now a 40x40 array with raw elevation values in meters")
    print("   # 0 = sea level/ocean, positive values = elevation above sea level")
    
    return elevation_40x40_raw, metadata


if __name__ == "__main__":
    elevation_raw, meta = main()
