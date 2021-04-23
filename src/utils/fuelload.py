import xarray
import numpy as np

BA_THRESH = 7.7 * 1e6  # Threshold for burned area
AGB_THRESH = 0  # Threshold for AGB dataset


def fuelload(path_agb: str, path_ba: str, path_fl: str = None) -> xarray.Dataset:
    """This function generates Fuel Load xarray dataset

    Parameters
    -----------
    path_agb : str
        path of AGB dataset
    path_ba : str
        path of BA  dataset
    path_fl : str
        path to save final fuel load dataset. Defaults to None.

    Returns
    --------
    xarray.Dataset
        Returns xarray dataset of fuel load
    """
    da_agb = xarray.open_dataset(path_agb)
    da_ba = xarray.open_dataset(path_ba)
    agb_data = da_agb["abg_avitabile_vodmean"][:, :, :]
    ba_data = da_ba["burned_area"][:, :, :]
    agb_data.values[agb_data.values == AGB_THRESH] = np.nan
    ba_data.values[ba_data.values < BA_THRESH] = np.nan

    # AGB units are Mg/h
    # BA units are m2, therefore we convert BA to hectares.
    ba_data = ba_data * 0.0001
    
    # Now that units are consistent we calculate LOAD = AGB * BA
    fuel_load_dataset = agb_data * ba_data

    if path_fl is not None:
        fuel_load_dataset.to_netcdf(path_fl)

    return fuel_load_dataset
