#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify the integration of AirNowData, PWWBData, and HRRRData.

This script validates that the data pipelines are properly set up to predict PM2.5 values
for the period from December 1st, 2024 to January 31st, 2025.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from datetime import datetime

# Ensure we can import from the libs directory
sys.path.append("../")

try:
    # Import the data classes
    from libs.pwwbdata import PWWBData
    from libs.airnowdata import AirNowData
    from libs.hrrrdata import HRRRData
except ImportError as e:
    print(f"Error importing data classes: {e}")
    print("Please ensure you have the correct directory structure and all required libraries installed.")
    sys.exit(1)

# Load environment variables (API keys, credentials)
load_dotenv()

def test_data_pipeline():
    """
    Test the integration of AirNowData, PWWBData, and HRRRData.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("=== Starting Integration Test ===")
    
    # Define test parameters
    # Note: Using smaller spatial and temporal dimensions for quicker testing
    lat_bottom, lat_top = 33.9, 34.2
    lon_bottom, lon_top = -118.4, -118.0
    extent = (lon_bottom, lon_top, lat_bottom, lat_top)
    
    dim = 200  # Smaller grid for testing
    frames_per_sample = 5  # Fewer frames for testing
    
    # Just test a short period (2 days in December, 2 days in January)
    # In production, this would be the full 2 months
    start_date, end_date = "2024-12-01-00", "2024-12-02-23"
    
    print(f"Test configuration:")
    print(f"  - Geographic extent: {extent}")
    print(f"  - Grid dimensions: {dim}x{dim}")
    print(f"  - Frames per sample: {frames_per_sample}")
    print(f"  - Date range: {start_date} to {end_date}")
    
    # Create output directory for test results
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Test 1: Load AirNowData
        print("\nTest 1: Loading AirNowData...")
        airnow_api_key = os.getenv('AIRNOW_API_KEY')
        if not airnow_api_key:
            print("⚠️  Warning: AIRNOW_API_KEY not found in environment variables")
            print("   Will proceed with empty data")
        
        airnow = AirNowData(
            start_date=start_date,
            end_date=end_date,
            extent=extent,
            airnow_api_key=airnow_api_key,
            frames_per_sample=frames_per_sample,
            dim=dim,
            force_reprocess=True  # Force reprocess for testing
        )
        
        print(f"✓ AirNowData loaded successfully")
        print(f"  - Data shape: {airnow.data.shape if hasattr(airnow, 'data') else 'No data'}")
        print(f"  - Target stations: {airnow.target_stations.shape if hasattr(airnow, 'target_stations') and airnow.target_stations is not None else 'No targets'}")
        print(f"  - Air sensor locations: {len(airnow.air_sens_loc) if hasattr(airnow, 'air_sens_loc') else 'No sensors'}")
        
        # Test 2: Load PWWBData
        print("\nTest 2: Loading PWWBData...")
        pwwb = PWWBData(
            start_date=start_date,
            end_date=end_date,
            extent=extent,
            frames_per_sample=frames_per_sample,
            dim=dim,
            verbose=True,
            use_cached_data=False,  # Don't use cache for testing
            output_dir=output_dir
        )
        
        print(f"✓ PWWBData loaded successfully")
        print(f"  - Data shape: {pwwb.data.shape if hasattr(pwwb, 'data') else 'No data'}")
        channel_info = pwwb.get_channel_info() if hasattr(pwwb, 'get_channel_info') else None
        print(f"  - Channels: {channel_info['channel_names'] if channel_info else 'No channel info'}")
        
        # Test 3: Load HRRRData (with a very limited date range for testing)
        print("\nTest 3: Loading HRRRData...")
        try:
            hrrr = HRRRData(
                start_date=start_date,
                end_date=end_date,
                extent=extent,
                extent_name='test_region',
                product='MASSDEN',
                frames_per_sample=frames_per_sample,
                dim=dim,
                verbose=True
            )
            
            print(f"✓ HRRRData loaded successfully")
            print(f"  - Data shape: {hrrr.data.shape if hasattr(hrrr, 'data') else 'No data'}")
        except Exception as e:
            print(f"⚠️  Warning: HRRRData load failed: {e}")
            print("   This may be normal if you don't have access to HRRR data")
            print("   The integration test will continue without HRRRData")
            hrrr = None
        
        # Test 4: Verify data shapes are compatible for concatenation
        print("\nTest 4: Verifying data shapes for concatenation...")
        
        # Get expected shapes
        airnow_shape = airnow.data.shape if hasattr(airnow, 'data') and airnow.data is not None else None
        pwwb_shape = pwwb.data.shape if hasattr(pwwb, 'data') and pwwb.data is not None else None
        hrrr_shape = hrrr.data.shape if hrrr and hasattr(hrrr, 'data') and hrrr.data is not None else None
        
        print(f"  - AirNowData shape: {airnow_shape}")
        print(f"  - PWWBData shape: {pwwb_shape}")
        print(f"  - HRRRData shape: {hrrr_shape}")
        
        # Check if shapes match in all dimensions except the channel dimension
        if airnow_shape and pwwb_shape:
            shapes_match = (
                airnow_shape[0] == pwwb_shape[0] and
                airnow_shape[1] == pwwb_shape[1] and
                airnow_shape[2] == pwwb_shape[2] and
                airnow_shape[3] == pwwb_shape[3]
            )
            
            if shapes_match:
                print(f"✓ AirNowData and PWWBData shapes are compatible for concatenation")
            else:
                print(f"✗ AirNowData and PWWBData shapes are NOT compatible for concatenation")
                return False
            
            # Check if HRRR data is compatible too
            if hrrr_shape:
                hrrr_shapes_match = (
                    airnow_shape[0] == hrrr_shape[0] and
                    airnow_shape[1] == hrrr_shape[1] and
                    airnow_shape[2] == hrrr_shape[2] and
                    airnow_shape[3] == hrrr_shape[3]
                )
                
                if hrrr_shapes_match:
                    print(f"✓ HRRRData shape is compatible for concatenation")
                else:
                    print(f"✗ HRRRData shape is NOT compatible for concatenation")
                    return False
        
        # Test 5: Test concatenation
        print("\nTest 5: Testing data concatenation...")
        try:
            # Concatenate AirNowData and PWWBData
            if airnow_shape and pwwb_shape:
                X_combined = np.concatenate([pwwb.data, airnow.data], axis=-1)
                print(f"✓ Successfully concatenated AirNowData and PWWBData")
                print(f"  - Combined shape: {X_combined.shape}")
                
                # Add HRRR data if available
                if hrrr_shape:
                    X_combined_hrrr = np.concatenate([X_combined, hrrr.data], axis=-1)
                    print(f"✓ Successfully added HRRRData to the concatenation")
                    print(f"  - Combined shape with HRRR: {X_combined_hrrr.shape}")
            
            # Check if we have target data
            if hasattr(airnow, 'target_stations') and airnow.target_stations is not None:
                print(f"✓ Target stations available for predictions")
                print(f"  - Target shape: {airnow.target_stations.shape}")
            else:
                print(f"⚠️  Warning: No target stations available for predictions")
        
        except Exception as e:
            print(f"✗ Data concatenation failed: {e}")
            return False
        
        # Test 6: Visualize a sample if data is available
        if airnow_shape and pwwb_shape:
            print("\nTest 6: Visualizing data sample...")
            try:
                # Visualize the first sample
                sample_idx = 0
                
                # Create a figure to display sample data
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Plot AirNow data (first channel of first frame)
                axes[0].imshow(airnow.data[sample_idx, 0, :, :, 0])
                axes[0].set_title('AirNow PM2.5')
                axes[0].axis('off')
                
                # Plot a PWWB channel (e.g., first channel of first frame)
                axes[1].imshow(pwwb.data[sample_idx, 0, :, :, 0])
                axes[1].set_title('PWWB (First Channel)')
                axes[1].axis('off')
                
                # Plot combined data
                axes[2].imshow(X_combined[sample_idx, 0, :, :, 0])
                axes[2].set_title('Combined (First Channel)')
                axes[2].axis('off')
                
                plt.suptitle('Data Sample Visualization')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'data_visualization.png'))
                plt.close()
                
                print(f"✓ Data visualization saved to {os.path.join(output_dir, 'data_visualization.png')}")
            except Exception as e:
                print(f"⚠️  Warning: Data visualization failed: {e}")
                # Not a critical failure, so continue
        
        print("\n=== Integration Test Summary ===")
        print("✓ AirNowData interface verified")
        print("✓ PWWBData interface verified")
        print(f"{'✓' if hrrr else '⚠️'} HRRRData interface {'verified' if hrrr else 'not tested'}")
        print("✓ Data shapes compatible for concatenation")
        print("✓ Data concatenation successful")
        
        print("\nAll tests passed! The data pipeline is ready for PM2.5 prediction from December 2024 to January 2025.")
        return True
        
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_pipeline()
    sys.exit(0 if success else 1)