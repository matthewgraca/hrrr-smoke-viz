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

def std_scale(data):
    mean = data.mean()
    var = data.var()
    return (data - mean) / math.sqrt(var)

def std_scaler(data):
    return StandardScaler().fit_transform(data.reshape(-1, 1)).reshape(data.shape)

start = time.time()
man_scaled_hrrr = std_scale(ds_np)
end = time.time()
elapsed1 = end - start

start = time.time()
skl_scaled_hrrr = std_scaler(ds_np)
end = time.time()
elapsed2 = end - start

print("Manually", elapsed1)
print(man_scaled_hrrr)
print("sklearn", elapsed2)
print(skl_scaled_hrrr)

print(np.all(np.isclose(man_scaled_hrrr, skl_scaled_hrrr, rtol=1e-5, atol=1e-8)))

# results: manually is more readable and faster than sklearn's std scaler
