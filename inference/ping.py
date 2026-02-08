import requests
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('AIRNOW_API_KEY')
EXTENT = (-118.615, -117.70, 33.60, 34.35)
BBOX = f'{EXTENT[0]},{EXTENT[2]},{EXTENT[1]},{EXTENT[3]}'

URL = "https://www.airnowapi.org/aq/data"

def check_for_current_hour():
    now = datetime.utcnow()
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    
    start = current_hour - timedelta(hours=2)
    end = current_hour + timedelta(hours=1)
    
    params = {
        'startDate': start.strftime('%Y-%m-%dT%H'),
        'endDate': end.strftime('%Y-%m-%dT%H'),
        'parameters': 'PM25',
        'BBOX': BBOX,
        'dataType': 'B',
        'format': 'application/json',
        'verbose': '1',
        'monitorType': '2',
        'includerawconcentrations': '1',
        'API_KEY': API_KEY
    }
    
    response = requests.get(URL, params=params)
    data = response.json()
    
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        if 'WebServiceError' in data[0]:
            print(f"API Error: {data[0]['WebServiceError']}")
            return current_hour, None, 0, set()
    
    hours_found = set()
    for record in data:
        if 'UTC' in record:
            record_time = datetime.fromisoformat(record['UTC'].replace('Z', '+00:00')).replace(tzinfo=None)
            hours_found.add(record_time.hour)
    
    latest_hour = None
    for record in data:
        if 'UTC' in record:
            record_time = datetime.fromisoformat(record['UTC'].replace('Z', '+00:00')).replace(tzinfo=None)
            if latest_hour is None or record_time > latest_hour:
                latest_hour = record_time
    
    return current_hour, latest_hour, len(data), hours_found

print("Polling AirNow every minute until current hour is available...")
print("=" * 60)

while True:
    current_hour, latest_hour, record_count, hours_found = check_for_current_hour()
    now = datetime.utcnow()
    
    hours_str = sorted(hours_found) if hours_found else []
    print(f"[{now.strftime('%H:%M:%S')} UTC] Target: {current_hour.strftime('%H:00')} | Latest: {latest_hour.strftime('%H:00') if latest_hour else 'None'} | Hours found: {hours_str} | Records: {record_count}")
    
    if latest_hour and latest_hour.hour == current_hour.hour:
        print("\n✅ Current hour is now available!")
        print(f"   Posted at approximately {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"   Delay: {now.minute} minutes after the hour")
        break
    
    time.sleep(60)