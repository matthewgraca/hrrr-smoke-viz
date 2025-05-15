import os
import requests
import json
from datetime import datetime, timedelta
from pprint import pprint
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_modis_collections(search_term, date_range, geographic_bounds=None):
    """
    Test searching for MODIS collections and granules
    
    Parameters:
    -----------
    search_term : str
        Search term to find collections (e.g., 'MOD11A1', 'MOD14')
    date_range : tuple
        (start_date, end_date) in YYYY-MM-DD format
    geographic_bounds : tuple, optional
        (min_lon, max_lon, min_lat, max_lat)
    """
    # Get Earthdata token from environment
    earthdata_token = os.getenv('EARTHDATA_TOKEN')
    if not earthdata_token:
        print("Error: EARTHDATA_TOKEN not found in environment variables")
        return
    
    # Set up headers with bearer token
    headers = {"Authorization": f"Bearer {earthdata_token}"}
    
    # Search for collections
    print(f"\n=== Searching for collections matching '{search_term}' ===")
    collection_url = "https://cmr.earthdata.nasa.gov/search/collections.json"
    collection_params = {
        "keyword": search_term,
        "page_size": 10
    }
    
    try:
        response = requests.get(collection_url, params=collection_params, headers=headers)
        if response.status_code != 200:
            print(f"Error searching collections: {response.status_code}")
            print(f"Response: {response.text}")
            return
        
        collections = response.json().get('feed', {}).get('entry', [])
        if not collections:
            print(f"No collections found matching '{search_term}'")
            return
        
        print(f"Found {len(collections)} collections:")
        for i, collection in enumerate(collections):
            concept_id = collection.get('id')
            short_name = collection.get('short_name')
            version = collection.get('version_id', 'unknown')
            title = collection.get('title')
            
            print(f"\n{i+1}. {short_name} v{version} ({concept_id})")
            print(f"   Title: {title}")
            
            # Test granule search for this collection
            if date_range:
                test_granules_for_collection(concept_id, short_name, date_range, geographic_bounds, headers)
    
    except Exception as e:
        print(f"Error during collection search: {e}")

def test_granules_for_collection(concept_id, short_name, date_range, geographic_bounds, headers):
    """Test granule search for a specific collection"""
    start_date, end_date = date_range
    
    print(f"\n  Testing granule search for {short_name} ({concept_id})")
    print(f"  Date range: {start_date} to {end_date}")
    
    granule_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
    granule_params = {
        "collection_concept_id": concept_id,
        "temporal": f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
        "page_size": 5
    }
    
    # Add bounding box if provided
    if geographic_bounds:
        min_lon, max_lon, min_lat, max_lat = geographic_bounds
        granule_params["bounding_box"] = f"{min_lon},{min_lat},{max_lon},{max_lat}"
        print(f"  Geographic bounds: {min_lon},{min_lat},{max_lon},{max_lat}")
    
    try:
        response = requests.get(granule_url, params=granule_params, headers=headers)
        if response.status_code != 200:
            print(f"  Error searching granules: {response.status_code}")
            print(f"  Response: {response.text}")
            return
        
        granules = response.json().get('feed', {}).get('entry', [])
        if not granules:
            print(f"  No granules found for {short_name} in the specified date range")
            return
        
        print(f"  Found {len(granules)} granules for {short_name}:")
        for i, granule in enumerate(granules[:2]):  # Show details for first 2 granules
            granule_id = granule.get('id')
            title = granule.get('title')
            print(f"    {i+1}. {title}")
            print(f"       ID: {granule_id}")
            
            # Print download URL if available
            download_url = next((link["href"] for link in granule.get("links", []) 
                               if link.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#"), None)
            if download_url:
                print(f"       Download URL available")
            else:
                print(f"       No download URL found")
        
        if len(granules) > 2:
            print(f"    ... and {len(granules) - 2} more granules")
    
    except Exception as e:
        print(f"  Error during granule search: {e}")

def main():
    # Geographic bounds for Los Angeles County (approximately)
    la_bounds = (-118.75, -117.5, 33.5, 34.5)
    
    # Test with different date ranges
    # Try a historical date range (definitely should have data)
    historical_dates = ("2020-07-01", "2020-07-02")
    
    # Try a more recent date range
    recent_dates = ("2023-07-01", "2023-07-02")
    
    # Try the current/future dates
    current_dates = ("2024-12-01", "2024-12-02")
    
    # Test different MODIS products
    print("\n===== Testing MODIS Land Surface Temperature Product =====")
    test_modis_collections("MOD11A1", historical_dates, la_bounds)
    
    print("\n===== Testing MODIS Fire Product =====")
    test_modis_collections("MOD14", historical_dates, la_bounds)
    
    print("\n===== Testing with Recent Dates =====")
    test_modis_collections("MOD11A1", recent_dates, la_bounds)
    test_modis_collections("MOD14", recent_dates, la_bounds)
    
    print("\n===== Testing with Current/Future Dates =====")
    test_modis_collections("MOD11A1", current_dates, la_bounds)
    test_modis_collections("MOD14", current_dates, la_bounds)

if __name__ == "__main__":
    main()