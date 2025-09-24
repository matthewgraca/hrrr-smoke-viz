import json
import numpy as np
import pandas as pd
import sys

class NowCastChecker:
    
    def calculate_nowcast(self, hourly_concentrations):
        df = pd.DataFrame({'concentration': hourly_concentrations, 
                          'hours_ago': range(len(hourly_concentrations))})
        
        recent_3 = df.head(3)['concentration']
        if recent_3.notna().sum() < 2:
            return None
        
        valid_df = df[df['concentration'].notna()]
        if len(valid_df) < 2:
            return None
        
        # Step 1: Select the minimum and maximum PM measurements
        c_min = valid_df['concentration'].min()
        c_max = valid_df['concentration'].max()

        # Step 2: Subtract the minimum measurement from the maximum measurement to get the range
        c_range = c_max - c_min

        # Step 3: Divide the range by the maximum measurement in the 12 hour period to get the scaled rate of change
        if c_max > 0:
            scaled_rate_of_change = c_range / c_max
        else:
            scaled_rate_of_change = 0

        # Step 4: Subtract the scaled rate of change from 1 to get the weight factor
        weight_factor = 1 - scaled_rate_of_change
        
        # Step 5: If the weight factor is less than 0.5, then set it equal to 0.5
        weight_factor = max(weight_factor, 0.5)
        
        # Step 6: Multiply each hourly measurement by the weight factor raised to the power of the number of hours ago
        df['weight'] = df['hours_ago'].apply(lambda x: weight_factor ** x)
        df['weighted_conc'] = df['concentration'] * df['weight']
        
        # Step 7: Compute the NowCast by summing the products from Step 6 and dividing by the sum of the weight factor
        valid_weighted = df[df['concentration'].notna()]
        numerator = valid_weighted['weighted_conc'].sum()
        denominator = valid_weighted['weight'].sum()
        
        if denominator > 0:
            nowcast = numerator / denominator
            return nowcast
        else:
            return None


def process_data(json_file_path):
    print(f"Loading data from {json_file_path}...")
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    df['RawConcentration'] = df['RawConcentration'].replace(-999.0, np.nan)
    df['Value'] = df['Value'].replace(-999.0, np.nan)
    
    df['datetime'] = pd.to_datetime(df['UTC'])
    
    df = df[df['SiteName'] != 'N/A']
    
    df = df.sort_values(['SiteName', 'datetime'])
    
    date_cutoff = df['datetime'].max() - pd.Timedelta(days=30)
    df = df[df['datetime'] >= date_cutoff]
    
    print(f"Processing {len(df)} records from {len(df['SiteName'].unique())} sites")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    checker = NowCastChecker()
    
    results = []
    
    for site_name in df['SiteName'].unique():
        site_data = df[df['SiteName'] == site_name].sort_values('datetime').reset_index(drop=True)
        
        if len(site_data) < 12:
            continue
        
        for i in range(11, len(site_data)):
            hourly_data = []
            for j in range(12):
                idx = i - j
                if idx >= 0:
                    conc = site_data.iloc[idx]['RawConcentration']
                    hourly_data.append(conc if pd.notna(conc) else None)
                else:
                    hourly_data.append(None)
            
            calculated = checker.calculate_nowcast(hourly_data)
            
            if calculated is not None:
                calculated_rounded = round(calculated, 1)
                
                provided = site_data.iloc[i]['Value']
                
                if pd.notna(provided):
                    results.append({
                        'site': site_name,
                        'datetime': site_data.iloc[i]['datetime'],
                        'raw_concentration': site_data.iloc[i]['RawConcentration'],
                        'calculated_nowcast': calculated_rounded,
                        'provided_nowcast': provided,
                        'difference': calculated_rounded - provided
                    })
    
    return pd.DataFrame(results)


def analyze_results(results_df):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.2f}'.format)
    
    print("\n" + "="*60)
    print("NOWCAST COMPARISON RESULTS")
    print("="*60)
    
    print(f"\nTotal comparisons: {len(results_df)}")
    
    exact_matches = (results_df['difference'] == 0.0).sum()
    exact_pct = 100 * exact_matches / len(results_df)
    
    print(f"\nExact matches: {exact_matches} ({exact_pct:.1f}%)")
    print(f"Non-exact: {len(results_df) - exact_matches} ({100-exact_pct:.1f}%)")
    
    print("\nDifference Statistics:")
    stats_data = {
        'Metric': ['Mean', 'Std Dev', 'Min', 'Max'],
        'Value (µg/m³)': [
            results_df['difference'].mean(),
            results_df['difference'].std(),
            results_df['difference'].min(),
            results_df['difference'].max()
        ]
    }
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))
    
    print("\nAccuracy within thresholds:")
    threshold_data = []
    for threshold in [10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.01, 0.0]:
        within = (results_df['difference'].abs() <= threshold).sum()
        pct = 100 * within / len(results_df)
        threshold_data.append({
            'Threshold (±µg/m³)': threshold,
            'Count': within,
            'Percentage': f"{pct:.1f}%"
        })
    threshold_df = pd.DataFrame(threshold_data)
    print(threshold_df.to_string(index=False))
    
    mismatches = results_df[results_df['difference'] != 0.0]
    if len(mismatches) > 0:
        print("\nExample mismatches (first 10):")
        display_cols = ['site', 'datetime', 'calculated_nowcast', 'provided_nowcast', 'difference']
        print(mismatches[display_cols].head(10).to_string(index=False))
    
    print("\nSite-specific exact match rates:")
    site_stats = results_df.groupby('site').agg({
        'difference': lambda x: (x == 0.0).mean() * 100
    }).round(1)
    site_stats.columns = ['Exact_Match_%']
    site_stats = site_stats.sort_values('Exact_Match_%')
    print(site_stats.to_string())
    
    return results_df


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_nowcast.py <json_file_path>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    try:
        results_df = process_data(json_file)
        
        if len(results_df) == 0:
            print("No valid comparisons found!")
            return
        
        analyze_results(results_df)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()