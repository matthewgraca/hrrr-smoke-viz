import numpy as np
import rasterio
from rasterio.windows import from_bounds
from scipy import ndimage
import matplotlib.pyplot as plt

def load_data_convert_grid(tiff_file, lat_bottom, lat_top, lon_bottom, lon_top, target_size=(40, 40)): 
    with rasterio.open(tiff_file) as src:
        window = from_bounds(lon_bottom, lat_bottom, lon_top, lat_top, src.transform)
        data = src.read(1, window=window)
        nodata = src.nodata

    if nodata is not None:
        data[(data == nodata) | (data < -100)] = 0
    else:
        data[data < -100] = 0

    zoom_factors = (target_size[0] / data.shape[0], target_size[1] / data.shape[1])
    resampled = ndimage.zoom(data, zoom_factors, order=1)[:target_size[0], :target_size[1]]

    if resampled.shape != target_size:
        pad_h, pad_w = target_size[0] - resampled.shape[0], target_size[1] - resampled.shape[1]
        resampled = np.pad(resampled, ((0, pad_h), (0, pad_w)), mode='edge')

    return resampled


def visualize(data, target_size, output_path):
    plt.figure(figsize=(8, 8))
    plt.imshow(data, cmap='terrain')
    plt.title(f'Elevation ({target_size[0]}x{target_size[1]}) | Range: [{data.min():.0f}, {data.max():.0f}]m')
    plt.colorbar(label='Elevation (m)')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    tiff_file = '/mnt/wildfire/raw-data/aster-gdem/ASTGTM_NC.003_ASTER_GDEM_DEM_doy2000061000000_aid0001.tif'
    lat_bottom, lat_top = 33.60, 34.35
    lon_bottom, lon_top = -118.615, -117.70
    target_size = (84, 84)

    elevation = load_data_convert_grid(tiff_file, lat_bottom, lat_top, lon_bottom, lon_top, target_size)

    print(f"Shape: {elevation.shape} | Range: [{elevation.min():.0f}, {elevation.max():.0f}]m | Mean: {elevation.mean():.0f}m")

    out_name = f'elevation'
    np.save(f'{out_name}.npy', elevation)
    print(f"Saved: {out_name}.npy")

    visualize(elevation, target_size, f'{out_name}.png')

    return elevation


if __name__ == "__main__":
    elevation = main()
