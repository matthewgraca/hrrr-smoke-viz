import s3fs
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

class NAQFCData:
    def __init__(
        self,
        start_date="2025-01-10 00:00",
        end_date="2025-01-10 00:59",
        extent=(-118.75, -117.0, 33.5, 34.5),
        dim=40,
    ):
        s3 = s3fs.S3FileSystem(anon=True)
        print([filename.removeprefix('noaa-nws-naqfc-pds/') for filename in s3.ls('noaa-nws-naqfc-pds')])

        # example
        print([filename for filename in s3.ls('noaa-nws-naqfc-pds/AQMv7/CS/20240514/06') if 'ave_1hr_pm25_bc' in filename])

        # filepath structure
        # noaa-nws-naqfc-pds/AQMv7/CS/20240514/06/aqm.t06z.ave_1hr_pm25_bc.20240514.227.grib2
        # noaa-nws-naqfc-pds/{model}/{area}/{date}/{forecast_start}/aqm.t{forecast_start}z.ave_1hr_pm25_bc.{date}.{location}.grib2

        # here we can define the valid params
        area_code = {
            'AK' : 198,
            'CS' : 227,
            'HI' : 196
        }
        aqm_file_params = {
            'model' : {'AQMv5', 'AQMv6', 'AQMv7', 'AQMv7_suppl'},
            'area' : {'AK', 'CS', 'HI'},
            'forecast_start' : {'06', '12'}
        }

        # then construct a valid set of params
        path_params = {
            'model' : 'AQMv7',
            'area' : 'CS',
            'forecast_start' : '06',
            'date' : '20240514'
        }
        print(aqm_file_params)
        path = '/'.join([
            f'noaa-nws-naqfc-pds',
            f'{path_params["model"]}',
            f'{path_params["area"]}',
            f'{path_params["date"]}',
            f'{path_params["forecast_start"]}',
            f'aqm.t{path_params["forecast_start"]}z.ave_1hr_pm25_bc.{path_params["date"]}.{area_code[path_params["area"]]}.grib2'
        ])
        print(path)
        src = s3.ls(path)[0]
        print(src)
        '''
        broadly speaking, there are 3 classes
        1. pm2.5 vs ozone
        2. bias corrected vs original
            - we'll use bias corrected for now until we have a good reason not to
        3. x-hour average, x-hour max
            - we want 1-hour average
            - 24-hour average is just the average value over 24 hours (only gives 3 values)
            - 1-hour max just gives the maximum value over the 72 hours (only gives 1 value)
        So we want: 'ave_1hr_pm25_bc'
        '''

        # open and read data
        '''
        s3.get(src, '/home/mgraca/Workspace/hrrr-smoke-viz')
        '''
        # does not like with open() for some reason, some io buffer subscripting error
        ds = xr.load_dataset('/home/mgraca/Workspace/hrrr-smoke-viz/tests/naqfcdata/data/aqm.t06z.ave_1hr_pm25_bc.20240514.227.grib2', engine='cfgrib')
    
        print(ds)
        '''
        Probably gonna be s3.open(...) with xarray open/load dataset with cfgrib engine
        Look into backend_kwargs to see if you can filter things by keys to trim the file
        with s3.open(s3.ls(path)[0], 'rb') as f:
            print(f'opening {f}...')
            ds = xr.open_dataset(f, engine='cfgrib')
            print('complete.')
            print(ds.info())
        '''


        # plottingz -> see jupyter notebook
