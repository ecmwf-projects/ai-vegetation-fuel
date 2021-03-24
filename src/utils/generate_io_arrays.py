import os
import xarray as xr
import pandas as pd
from typing import Tuple


from utils.data_paths import export_feature_paths

LIST_DF_NAMES = [
    "train.csv",
    "val.csv",
    "test.csv",
]  # Filenames for test train val dataframes


def data_split(data: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """This generates split dataset with merged input and output values.
    Currently the data is split in the ratio 12:1:1 (train:validation:test) as follows -
    Training: 2010 -> 2015
    Validation: January 2016 -> June 2016
    Testing: July 2016 -> December 2016.

    Parameters
    -----------
    data : xr.Dataset
        Combined xarray dataset of input and output

    Returns
    --------
    Tuple[xr.Dataset, xr.Dataset, xr.Dataset]
        Returns tuple with training,validation and test xarray datasets.
    """
    data_train = data.sel(
        time=slice(data.time.values[0], data.time.values[71])
    )  # The indicies 0 and 71 corresponds to the 72 months for train split

    data_val = data.sel(
        time=slice(data.time.values[72], data.time.values[77])
    )  # The indicies 72 and 77 corresponds to the 6 months for val split

    data_test = data.sel(
        time=slice(data.time.values[78], data.time.values[-1])
    )  # The indicies 78 and -1 corresponds to the 6 months for test split

    return data_train, data_val, data_test


def data_combined(path: str, path_fuelload: str, path_save: str = None) -> xr.Dataset:
    """This creates combined xarray dataset with output values

    Parameters
    -----------
    path : str
        path of root folder where all individual datasets are stored
    path_fuelload : str
        path in which combined xarray dataset is to be saved
    path_save : str
        path of  fuel load dataset. Defaults to None.

    Returns
    --------
    xr.Dataset
        Returns combined xarray dataset with input and output variables
    """
    # Process to combine all the datasets
    list_files = os.listdir(path)
    list_data = []

    # import datafile paths
    time_dependant_features, time_independant_features = export_feature_paths()

    for filename in list_files:
        if filename in time_dependant_features:
            list_data.append(
                xr.open_dataset(
                    os.path.join(path, filename),
                )
            )
        combined_dataset = xr.merge(list_data)

    # handle static variables
    for filename in time_independant_features:
        feature_dataset = xr.open_dataset(os.path.join(path, filename))
        feature_dataset = feature_dataset.expand_dims(
            {"time": combined_dataset.time.values}
        )
        combined_dataset = combined_dataset.merge(feature_dataset)

    output = xr.open_dataset(
        path_fuelload,
    )
    output = output.rename_vars({"__xarray_dataarray_variable__": "actual_load"})
    combined_dataset = combined_dataset.merge(output)

    if path_save is not None:
        combined_dataset.to_netcdf(path_save)

    return combined_dataset


# Conversion to dataframe
def data_to_narray(
    dataset: xr.Dataset, path_df: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """This creates pandas dataframe of train ,validation and test data

    Parameters
    -----------
    dataset : xr.Dataset
        Combined xarray dataset
    path_df : str
        Path where train,test,val dataframe as CSV is to be saved. Defaults to None.

    Returns
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Returns a tuple with training,validation,test dataframes.
    """
    data_train, data_val, data_test = data_split(dataset)

    list_df = [data_train, data_val, data_test]
    list_df_names = LIST_DF_NAMES
    list_df_processed = []

    for i, df in enumerate(list_df):
        df = (
            df.to_dataframe()
            .reset_index(["time", "latitude", "longitude"])
            .dropna()
            .reset_index(drop=True)
        )
        df["month"] = pd.DatetimeIndex(df["time"]).month
        list_df_processed.append(df)

        if path_df is not None:
            df.to_csv(os.path.join(path_df, list_df_names[i]), index=False)

    return list_df_processed[0], list_df_processed[1], list_df_processed[2]
