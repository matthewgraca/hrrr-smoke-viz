import xarray as xr
import time
from contextlib import redirect_stdout

start = time.perf_counter()
path = '/mnt/0CA6310DA630F932/goes_cache/noaa-goes18/ABI-L2-AODC/2023/212/23'
files = [
    f'{path}/OR_ABI-L2-AODC-M6_G18_s20232122301185_e20232122303558_c20232122306252.nc',
    f'{path}/OR_ABI-L2-AODC-M6_G18_s20232122331185_e20232122333558_c20232122336051.nc',
    f'{path}/OR_ABI-L2-AODC-M6_G18_s20232122306185_e20232122308558_c20232122311058.nc',
    f'{path}/OR_ABI-L2-AODC-M6_G18_s20232122336185_e20232122338558_c20232122341170.nc',
    f'{path}/OR_ABI-L2-AODC-M6_G18_s20232122311185_e20232122313558_c20232122316097.nc',
    f'{path}/OR_ABI-L2-AODC-M6_G18_s20232122341185_e20232122343558_c20232122345556.nc',
    f'{path}/OR_ABI-L2-AODC-M6_G18_s20232122316185_e20232122318558_c20232122321038.nc',
    f'{path}/OR_ABI-L2-AODC-M6_G18_s20232122346185_e20232122348558_c20232122350575.nc',
    f'{path}/OR_ABI-L2-AODC-M6_G18_s20232122321185_e20232122323558_c20232122326172.nc',
    f'{path}/OR_ABI-L2-AODC-M6_G18_s20232122351185_e20232122353558_c20232122355516.nc',
    f'{path}/OR_ABI-L2-AODC-M6_G18_s20232122326185_e20232122328558_c20232122331073.nc',
    f'{path}/OR_ABI-L2-AODC-M6_G18_s20232122356185_e20232122358558_c20232130001032.nc',
]

for file in files:
    xr.load_dataset(file)

end = time.perf_counter()
print(f"Manual: {end-start:.4} seconds elapsed")

from goes2go.data import goes_timerange
import io

start = time.perf_counter()

with redirect_stdout(io.StringIO()):
    g = goes_timerange(
        start='2024-07-31T00:00:00',
        end='2024-07-31T00:59:59',
        satellite= 'goes18',
        product= 'ABI-L2-AODC',
        return_as= 'xarray',
        max_cpus= None,
        verbose = False,
        ignore_missing = False,
        save_dir='/mnt/0CA6310DA630F932/goes_cache', 
        download=False
    )
end = time.perf_counter()
print(f"Multithread: {end-start:.4} seconds elapsed")

#NOTE
'''
Suspicious, it takes 3.4 seconds to single-threaded load the files for a frame,
yet it takes the same time to do it multithreaded? Looking at htop, every thread
does get worked, so I'm not sure why this is the case.

I could reimplement the multithreading here to test.
'''

