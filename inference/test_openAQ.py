import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

from libs.openaqdata import OpenAQData

now = pd.Timestamp.now(tz='UTC').floor('h').tz_localize(None)
input_start = now - pd.Timedelta(hours=23)

print(f"start: {input_start.strftime('%Y-%m-%d %H:%M')}")
print(f"end:   {now.strftime('%Y-%m-%d %H:%M')}")

openaq = OpenAQData(
    api_key=os.getenv('OPENAQ_API_KEY'),
    start_date=input_start.strftime('%Y-%m-%d %H:%M'),
    end_date=now.strftime('%Y-%m-%d %H:%M'),
    extent=(-118.615, -117.70, 33.60, 34.35),
    dim=84,
    elevation_path='data/elevation.npy',
    save_dir='/tmp/test_openaq',
    save_path='/tmp/test_openaq/processed.npz',
    verbose=0,
)

print(f"Output shape: {openaq.data.shape}")