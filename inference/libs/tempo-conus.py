"""
Test TEMPO data latency - check V04 Standard vs NRT
"""
import earthaccess
from datetime import datetime, timedelta
import pandas as pd

earthaccess.login()

now = datetime.utcnow()
print(f"Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

products = [
    # L2 products
    ("TEMPO_NO2_L2", "L2 NO2 V04"),
    ("TEMPO_NO2_L2_NRT", "L2 NO2 NRT"),
    ("TEMPO_RAD_L1", "L1 Radiance"),  # Check L1 too
    # L3 for comparison
    ("TEMPO_NO2_L3", "L3 NO2 V04"),
    ("TEMPO_NO2_L3_NRT", "L3 NO2 NRT"),
]

for short_name, description in products:
    print(f"\n{description}")
    print(f"Short name: {short_name}")
    print("-" * 50)
    
    start = now - timedelta(days=7)
    
    try:
        granules = earthaccess.search_data(
            short_name=short_name,
            temporal=(start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")),
            count=20
        )
        
        print(f"Found {len(granules)} granules in last 7 days")
        
        if granules:
            granule_times = []
            for g in granules:
                umm = g.get('umm', {})
                temporal = umm.get('TemporalExtent', {})
                range_dt = temporal.get('RangeDateTime', {})
                end_time = range_dt.get('EndingDateTime')
                
                if end_time:
                    granule_times.append({
                        'end_time': pd.to_datetime(end_time),
                        'granule_id': g['meta'].get('native-id', 'N/A')
                    })
            
            granule_times.sort(key=lambda x: x['end_time'], reverse=True)
            
            if granule_times:
                most_recent = granule_times[0]
                latest_time = most_recent['end_time'].to_pydatetime().replace(tzinfo=None)
                latency = now - latest_time
                latency_hours = latency.total_seconds() / 3600
                
                print(f"\nMost recent data: {latest_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                print(f"Current time:     {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                print(f">>> LATENCY: {latency_hours:.1f} hours ({latency_hours/24:.1f} days) <<<")
                
                print(f"\nLast 10 granules:")
                for i, gt in enumerate(granule_times[:10]):
                    t = gt['end_time']
                    age = (now - t.to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600
                    print(f"  {i+1}. {t.strftime('%Y-%m-%d %H:%M')} UTC ({age:.1f}h ago)")
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)