from libs.goesdata import GOESData
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio.v3 as iio
import imageio
import numpy as np
import glob

path = "aod_images/idw_top_2_quality"
gd = GOESData(start_date="2025-01-16-12", end_date="2025-01-17-00", save_cache=True, cache_path='/home/mgraca/Downloads/goes_processed.npz')
print(gd.data.shape)
'''
for i, data in tqdm(enumerate(gd.data)):
    plt.imshow(data)
    plt.savefig(f'{path}/frame_{i:03d}.png')

# Get a list of image files in order
filenames = sorted(glob.glob(f"{path}/frame_*.png"))

# Create the GIF
iio.imwrite(f"{path}/animation.gif", [iio.imread(f) for f in filenames], duration=300) # duration in milliseconds

#Create reader object for the gif
paths = ["all_quality", "low_quality", "top_2_quality", "idw_top_2_quality"]
gif_files = [sorted(glob.glob(f"aod_images/{p}/frame_*.png")) for p in paths]

#Create writer object
with imageio.get_writer('aod_images/output.gif', duration=300) as new_gif:
    for frame_number in range(len(gif_files[0])):
        imgs = [iio.imread(gif_file[frame_number]) for gif_file in gif_files]
        #here is the magic
        new_image = np.hstack(imgs)
        new_gif.append_data(new_image)
'''
