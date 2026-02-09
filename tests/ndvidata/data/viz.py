from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt

f = SD('MOD13A2.A2023209.h08v05.061.2023226000837.hdf', SDC.READ)
ds = f.select('1 km 16 days NDVI')
data = ds[:]
plt.imshow(data); plt.show()

