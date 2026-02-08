"""
PWWB: Predict What We Breathe

This package provides tools to collect and process meteorological and remote sensing data
for environmental analysis, with a focus on air quality and atmospheric conditions.

Main Components:
- PWWBData: The main class for data collection and processing
- Data Sources: Individual data source modules for MAIAC, TROPOMI, MODIS, MERRA-2, and METAR data
- Utilities: Helper functions for caching, interpolation, and grid operations
"""

from libs.pwwb.pwwb_data import PWWBData

__version__ = "0.1.0"
__all__ = ["PWWBData"]