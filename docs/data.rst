Data Description
=================

7 years of global historical data, from 2010 - 2016 will be used for
developing the machine learning models. All data used in this project is
propietary and NOT meant for public release. Xarray and netCDF libraries
are used for working with the multi-dimensional geospatial data.

- **Datasets**:
   - Above Ground Biomass
   - Weather Anomalies
   - Climatic Regions
   - Fire Sensitivity Anomalies
   - Slope
   - Fraction of Burnable Area
   - Burned Area
   - Standardized Precipitation Index GPCC
   - Leaf Area Index
- **Resolution**:
   - Latitude: 0.25
   - Longitude: 0.25
   - Time: 1 file per month, ie. 84 timesteps from 2010-16.

The data split into training, testing and validation is currently:
- Training: 2010 -> 2015
- Validation: January 2016 -> June 2016
- Testing: July 2016 -> December 2016.

To change the split, modify `data_split()` in `src/utils/generate_io_arrays.py`, and the month list in `src/test.py` used during inference.
