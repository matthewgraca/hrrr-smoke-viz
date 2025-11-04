import s3fs
import pandas as pd

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
        # noaa-nws-naqfc-pds/{model}/{area}/{date}/{forecast_start}
        aqm_file_params = {
            'model' : {'AQMv5', 'AQMv6', 'AQMv7', 'AQMv7_suppl'},
            'area' : {'AK', 'CS', 'HI'},
            'forecast_start' : {'06', '12'}
        }
        print(aqm_file_params)
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

