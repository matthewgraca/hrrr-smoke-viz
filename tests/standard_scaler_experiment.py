from sklearn.preprocessing import StandardScaler
from herbie import Herbie
import math
import numpy as np
import time

# dl data
day, hour = 10, 0
date = f"2025-01-{str(day).zfill(2)}-{str(hour).zfill(2)}"
H = Herbie(
    date,
    model="hrrr",
    product="sfc",
)

# grib -> xarray -> np
ds = H.xarray("MASSDEN")
ds_np = ds.mdens.to_numpy()
print(ds_np.shape)

# standard scalers
def arr_std_scale(data):
    mean = data.mean()
    var = data.var()
    return (data - mean) / math.sqrt(var)

def skl_std_scale(data):
    return StandardScaler().fit_transform(data.reshape(-1, 1)).reshape(data.shape)

def np_std_scale(data):
    return (data - np.mean(data)) / np.std(data)

# perform experiments
results = {}
start = time.time()
man_scaled_hrrr = arr_std_scale(ds_np)
end = time.time()
results['array'] = end - start

start = time.time()
skl_scaled_hrrr = skl_std_scale(ds_np)
end = time.time()
results['sklearn'] = end - start

start = time.time()
np_scaled_hrrr = np_std_scale(ds_np)
end = time.time()
results['numpy'] = end - start

# print results 
sorted_res = {k: v for k, v in sorted(results.items(), key=lambda val: val[1])}
print("Results ordered from fastest to slowest")
print(sorted_res, '\n')

print("Check if all three scaled datasets agree with each other")
print(np.all(np.isclose(man_scaled_hrrr, skl_scaled_hrrr, rtol=1e-5, atol=1e-8)))
print(np.all(np.isclose(skl_scaled_hrrr, np_scaled_hrrr, rtol=1e-5, atol=1e-8)))
