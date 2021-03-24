import numpy as np
import os
import pandas as pd
from typing import Tuple

pd.options.mode.chained_assignment = None

MONTH_LIST = [
    "July",
    "Aug",
    "Sept",
    "Oct",
    "Nov",
    "Dec",
]  # Month list to split test dataframe

# Tropics coordinates
TROPICS_NORTH = 30
TROPICS_SOUTH = -30

# Midlats coordinates
MIDLATS_NORTH_LOWER = 30
MIDLATS_NORTH_HIGHER = 80
MIDLATS_SOUTH_LOWER = -30
MIDLATS_SOUTH_HIGHER = -80


def _month_wise(x: str, model_type: str, data_dict: dict, path_df: str):
    """Helper function to generate month wise test files

    Parameters
    ----------
    x : str
        type of data(train/test/val)
    model_type : str
        Type of model can be either tropics or midlats
    data_dict : dict
        dictonary of datasets
    path_df : str
        path where the month wise splitted test dataframe as CSV is to be saved
    """
    if os.path.exists(os.path.join(path_df, "infer_" + model_type)) is not True:
        os.makedirs(os.path.join(path_df, "infer_" + model_type))
    month_list = MONTH_LIST
    if x == "test":
        for i in range(0, len(month_list)):
            name = model_type + "_infers_" + month_list[i] + ".csv"
            testdf = data_dict[x][
                (data_dict[x].month == i + 7)
            ]  # Since month column of dataframe consists of value 1-12, 7 corresponds to July

            testdf.to_csv(
                os.path.join(path_df, "infer_" + model_type, name), index=False
            )


def tropics_dataset(
    train: pd.core.frame.DataFrame,
    val: pd.core.frame.DataFrame,
    test: pd.core.frame.DataFrame,
    path_df: str = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # Tropics Data
    """Returns Splitted dataset with merged input and output values for tropics

    Note
    -----
    Tropics model is not using the Leaf Area Index feature.

    Parameters
    -----------
    train : pd.core.frame.DataFrame
        train dataframe
    val : pd.core.frame.DataFrame
        validation dataframe
    test : pd.core.frame.DataFrame
        test dataframe
    path_df : str
        path where train,test,val dataframe as CSV is to be saved.Defaults to None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Returns tuple with  training ,validation and test data as numpy arrays
    """
    data_dict = {"train": train, "val": val, "test": test}
    tropic_dict = {}
    if os.path.exists(os.path.join(path_df, "tropics")) is not True:
        os.makedirs(os.path.join(path_df, "tropics"))
    for x in data_dict:
        df = data_dict[x]
        df = df.drop("LAI", axis=1)
        tropic_dict[x] = df.loc[
            (df["latitude"] > TROPICS_SOUTH) & (df["latitude"] < TROPICS_NORTH)
        ]
        tropic_dict[x] = tropic_dict[x].drop("time", axis=1)
        _month_wise(x, "tropics", tropic_dict, path_df)  # Month wise test files

        if path_df is not None:
            tropic_dict[x].to_csv(
                os.path.join(path_df, "tropics", "tropics_" + x + ".csv"), index=False
            )

    return (
        np.array(tropic_dict["train"]),
        np.array(tropic_dict["val"]),
        np.array(tropic_dict["test"]),
    )


def midlats_dataset(
    train: pd.core.frame.DataFrame,
    val: pd.core.frame.DataFrame,
    test: pd.core.frame.DataFrame,
    path_df: str = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # Mid Lat data
    """Returns Splitted dataset with merged input and output values for midlats

    Note
    -----
    Midlats model is using month and climatic region as categorical inputs

    Parameters
    -----------
    train : pd.core.frame.DataFrame
        train dataframe
    val : pd.core.frame.DataFrame
        validation dataframe
    test : pd.core.frame.DataFrame
        test dataframe
    path_df : str
        path where train,test,val dataframe as CSV is to be saved.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Returns tuple with  training ,validation and test data as numpy arrays
    """
    data_dict = {"train": train, "val": val, "test": test}
    midlat_dict = {}
    if os.path.exists(os.path.join(path_df, "midlats")) is not True:
        os.makedirs(os.path.join(path_df, "midlats"))
    for x in data_dict:
        df = data_dict[x]
        midlat_dict[x] = df.loc[
            (
                (df["latitude"] > MIDLATS_SOUTH_HIGHER)
                & (df["latitude"] < MIDLATS_SOUTH_LOWER)
            )
            | (
                (df["latitude"] > MIDLATS_NORTH_LOWER)
                & (df["latitude"] < MIDLATS_NORTH_HIGHER)
            )
        ]
        midlat_dict[x]["month"] = pd.Series(midlat_dict[x]["month"]).astype(int)
        midlat_dict[x]["climatic_region"] = pd.Series(
            midlat_dict[x]["climatic_region"]
        ).astype(int)
        midlat_dict[x] = midlat_dict[x].drop("time", axis=1)
        _month_wise(
            x, "midlats", midlat_dict, path_df
        )  # Helper function to generate month wise test files

        if path_df is not None:
            midlat_dict[x].to_csv(
                os.path.join(path_df, "midlats", "midlats_" + x + ".csv"), index=False
            )

    return (
        np.array(midlat_dict["train"]),
        np.array(midlat_dict["val"]),
        np.array(midlat_dict["test"]),
    )
