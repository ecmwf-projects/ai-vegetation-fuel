"""Test script

This script is used to generate inference files and map plots.
Usage: :code:`python src/test.py model  ckpt testdir resultsdir`

Where:


:code:`model` is the ML model to be trained.

:code:`ckpt` is the directory which  contains the pre-trained model.

:code:`testdir` is the directory which  contains the test :code:`.csv` files to be used to generate inference files.

:code:`resultsdir` is the directory at which inference and map plots are to be saved.

The :code:`.png` images produced by this script are also stored in :code:`result_dir/` as :code:`{type}_month_predicted.png` or
:code:`{type}_month_actual.png` where :code:`type` can be :code:`tropics` or :code:`midlats`.


Note
-----

The :code:`.csv` files should be named as :code:`{type}_infers_{month}.csv`, where :code:`type`
can be :code:`tropics` or :code:`midlats`.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from utils.generate_map_plots import create_geometry, generate_plots
from joblib import load
import argparse

SCALER_FILENAME = "scaler.save"  # Save name for sklearn transform file


def inference(
    month: str,
    regr,
    df_test: pd.core.frame.DataFrame,
    transform: bool,
    path_df: str = None,
    test: bool = True,
):
    """This function generates inference files

    Parameters
    ----------
    month : str
        Corresponding month of the test file.
    regr :
        Trained ML model for inference generation
    df_test : pd.core.frame.DataFrame
        Test dataset used for inference generation
    transform : bool
        Whether to apply box-cox or not
    path_df : str
        Path to save the inference files. Defaults to None.
    test : bool
        If test files contains actual Fuel load values or not. Defaults to True.
    """

    df_test_pred = df_test
    if (
        test
    ):  # Condition for if the inference files contain true labels ,drop them from the dataframe to be used in prediction
        if transform:
            scaler = PowerTransformer(method="box-cox")
            scaler.fit_transform(np.array(df_test.actual_load).reshape(-1, 1))

        df_test_pred = df_test.drop(["actual_load"], axis=1)
    y_pred = regr.predict(df_test_pred)
    if test:
        if transform:
            y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        else:
            y_pred_inv = y_pred

        # If predicted fuel load values are below zero, using min-max normalization to change the prediction to the range of actual fuel load values
        if y_pred_inv.min() < 0:
            range_fl_predicted = max(y_pred_inv) - min(
                y_pred_inv
            )  # range of predicted fuel load values
            if range_fl_predicted != 0:
                y_pred_inv = (
                    y_pred_inv - min(y_pred_inv)
                ) / range_fl_predicted  # normalize predicted fuel load values based on its range
            range_fl_actual = max(df_test.actual_load) - min(df_test.actual_load)
            if range_fl_actual != 0:
                y_pred_inv = y_pred_inv * range_fl_actual + min(
                    df_test.actual_load
                )  # normalize predicted fuel load values based on actual fuel load range

        # Storing inference file as pandas dataframe
        output_df = pd.DataFrame(
            data={
                "lat": df_test.latitude,
                "lon": df_test.longitude,
                "actual_load": df_test.actual_load,
                "predicted_load": y_pred_inv,
                "APE": (
                    np.abs((df_test.actual_load - y_pred_inv) / df_test.actual_load)
                )
                * 100,
            }
        )
        print(
            "MAPE ",
            month,
            " :",
            np.mean(np.abs((df_test.actual_load - y_pred_inv) / df_test.actual_load))
            * 100,
        )

    else:
        scaler_filename = SCALER_FILENAME
        scaler = load(scaler_filename)  # Loading sklearn transformation
        if transform:
            y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        else:
            y_pred_inv = y_pred
        output_df = pd.DataFrame(
            data={
                "lat": df_test.latitude,
                "lon": df_test.longitude,
                "predicted_load": y_pred_inv,
            }
        )
    if path_df is not None:
        output_df.to_csv(path_df, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test model")
    parser.add_argument(
        "--model_name",
        metavar="n",
        choices=["CatBoost", "LightGBM"],
        help="Name of the model",
        required=True,
    )
    parser.add_argument(
        "--model_path", metavar="p", help="Path of the model", required=True
    )
    parser.add_argument(
        "--data_path", metavar="d", help="Path of the data files", required=True
    )
    parser.add_argument(
        "--results_path",
        metavar="r",
        help="Path to where results are to be stored",
        required=True,
    )

    args = parser.parse_args()

    model_name = args.model_name
    model_path = args.model_path
    datadir = args.data_path
    result_dir = args.results_path

    if (
        os.path.exists(os.path.join(result_dir)) is not True
    ):  # Create result storing directory if already not existing
        os.makedirs(os.path.join(result_dir))

    model = load(model_path)
    file_list = os.listdir(datadir)
    for csv_file_path in file_list:
        if csv_file_path.endswith(".csv"):

            # assumes file should be {type}_infers_{month}.csv format where 'type' = tropics or midlats
            output_file_path_pred = (
                result_dir
                + "/"
                + csv_file_path[0:7]
                + "_output_"
                + csv_file_path[15:-4]
                + ".csv"
            )  # extracting base file name

            month = csv_file_path[15:-4]  # extracting month name

            # function calls
            df = pd.read_csv(datadir + "/" + csv_file_path)
            if model_name == "CatBoost":
                transform = True
            else:
                transform = False

            if "actual_load" in df.columns:
                has_groundtruth = True
            else:
                has_groundtruth = False
            inference(
                month=month,
                regr=model,
                df_test=df,
                transform=transform,
                path_df=output_file_path_pred,
                test=has_groundtruth,
            )

    # Map plot generation
    file_list = os.listdir(result_dir)

    for csv_file_path in file_list:
        if csv_file_path.endswith(".csv"):

            # assumes file should be {type}_output_{month}.csv format where 'type' = tropics or midlats
            output_file_path_pred = (
                result_dir
                + "/"
                + csv_file_path[0:7]
                + "_"
                + csv_file_path[15:-4]
                + "_predicted.png"
            )  # extracting base file name and appending 'predicted'
            output_file_path_act = (
                result_dir
                + "/"
                + csv_file_path[0:7]
                + "_"
                + csv_file_path[15:-4]
                + "_actual.png"
            )  # extracting base file name and appending 'actual'
            month = csv_file_path[15:-4]  # extracting month name

            # function calls
            df = pd.read_csv(result_dir + "/" + csv_file_path)
            df_with_geometry, _ = create_geometry(df)
            generate_plots(
                df_with_geometry, output_file_path_act, output_file_path_pred, month
            )
