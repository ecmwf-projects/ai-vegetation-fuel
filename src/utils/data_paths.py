"""This script allows easy modification of features/files and their data paths or their versions.
"""
from typing import List, Tuple

# change file paths here
agb_data = "/agb_avitabile_2010-2016_v3.nc"
weather_anomalies_data = "weather_anomalies_2010-2016_v3.nc"
ba_data = "/ba_2010-2016_v3.nc"
ba_fraction_data = "ba_fraction_2010-2016_v3.nc"
fire_anomal_data = "fire_anomalies_2010-2016_v3.nc"
spi_gpcc_data = "spi-gpcc_2010-2016_v4.nc"
climate_reg_data = "climatic_regions_v3.nc"
slopes_data = "slopes_2010-2016_v3.nc"
lai_data = "LAI_monthly_2010-2016_v2.nc"

fuel_load_dataset = "/fuel_load.nc"
combined_dataset = "/combined_dataset.nc"


def export_feature_paths() -> Tuple[List[str], List[str]]:
    """This function exports paths of files of the features to be used.
    Features are split into 2 categories -
        * Time Dependant
        * Time Independant

    Returns
    --------
    Tuple[List[str], List[str]]
        Returns tuple of string with data paths for time dependent and independent features.
    """
    time_dependant_features = [
        weather_anomalies_data,
        ba_fraction_data,
        fire_anomal_data,
        spi_gpcc_data,
        lai_data,
    ]

    time_independant_features = [climate_reg_data, slopes_data]

    return time_dependant_features, time_independant_features


def export_data_paths() -> Tuple[str, str, str, str]:
    """This function exports paths of files of the features to be used along with path for datasets to be stored at.
        * Above Ground Biomass
        * Burned Area
        * Fuel Load
        * Combined Dataset

    Returns
    --------
    Tuple[str,str,str, str]
        Return tuple of string with export paths of AGB,Burned Area and combined dataset.
    """
    agb_data_path = agb_data
    ba_data_path = ba_data
    fuel_load_data_path = fuel_load_dataset
    combined_data_path = combined_dataset

    return agb_data_path, ba_data_path, fuel_load_data_path, combined_data_path
