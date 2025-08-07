from libs.goesdata import GOESData
import matplotlib.pyplot as plt

g = GOESData(
    start_date="2024-04-10-00",
    end_date="2024-04-17-00", 
    save_dir="/mnt/0CA6310DA630F932/goes_cache", 
    cache_path="/mnt/0CA6310DA630F932/goes_cache/testing_goes.npz", 
    save_cache=True,
    pre_downloaded=True,
    load_cache=False
)
print(g.data.shape)

'''
cache_path="/mnt/0CA6310DA630F932/goes_processed.npz" 
extent=(-118.75, -117.0, 33.5, 34.5)
start_date="2023-08-02-00"
end_date="2025-08-02-00" 

import numpy as np
from libs.pwwb.utils.dataset import sliding_window
for i in range(len(g.data)):
    if (g.data[i,:,:] == 0).all():
        print(i)
        g.data[i] = g.data[i-1] 

a, _ = sliding_window(np.expand_dims(g.data, -1), 5)

np.savez_compressed(
    cache_path,
    data=a,
    start_date=start_date,
    end_date=end_date,
    extent=extent
)
'''
'''
gd = GOESData(start_date="2023-08-06-14", end_date="2023-08-06-17", save_dir="/media/mgraca/Local Disk", cache_path="/media/mgraca/Local Disk/goes_processed.npz", save_cache=True, verbose=True)

for i, d in enumerate(gd.data):
    plt.imshow(d)
    plt.savefig(f'buh_{i}.png')
'''
