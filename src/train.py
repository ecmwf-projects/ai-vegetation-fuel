"""Training script

This script is used to run training modlues of CatBoost and LightGBM models and save the trained model files.

Usage: :code:`python src/train.py model <datadir/> exptname`

Where:

:code:`model` is the ML model to be trained.

:code:`datadir/` is the directory which  contains the train/val/test .csv files to be used to train the model.

The :code:`.joblib` files produced by this script are stored in :code:`src/results/pre-trained_models` as :code:`model.joblib` where model can be LightGBM or CatBoost.

:code:`exptname` is the name of the neptune experiment.

Note
-----

The :code:`.csv` files should contain the keyword :code:`train`, :code:`val` and :code:`test` in their respective file names.
No other :code:`.csv` files should contain the before mentioned keywords in their file names.
"""

import os
import pandas as pd
from models.catboost_module import CatBoost
from models.lightgbm_module import LightGBM
import neptune
from joblib import dump
import argparse

REPO = "ml-fuel"
LIST_FILE_NAMES = [
    "train",
    "val",
    "test",
]  # File names for test,train and val dataframes
NUM_ITERS = 2000  # NUM_ITERS is for number of boosting iterations

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument(
        "--model_name",
        metavar="n",
        choices=["CatBoost", "LightGBM"],
        help="Name of the model",
        required=True,
    )
    parser.add_argument(
        "--data_path", metavar="d", help="Path of the data files", required=True
    )
    parser.add_argument(
        "--exp_name", metavar="e", help="Experiment Name", required=True
    )
    args = parser.parse_args()

    model_name = args.model_name
    datadir = args.data_path
    exptname = args.exp_name

    file_list = os.listdir(datadir)
    dict_data = {}
    list_file_names = LIST_FILE_NAMES

    for csv_file_path in file_list:
        if csv_file_path.endswith(".csv"):
            name = [name for name in list_file_names if csv_file_path.find(name) > 0]
            dict_data[name[0]] = pd.read_csv(datadir + "/" + csv_file_path)

    neptune.init(
        api_token="ANONYMOUS",
        project_qualified_name="shared/step-by-step-monitoring-experiments-live",
    )
    print("Link for the created Neptune experiment--------")
    neptune.create_experiment(exptname)
    print("---------------------------------------")
    if model_name == "CatBoost":
        obj = CatBoost(dict_data["train"], dict_data["val"], dict_data["test"])
        model = obj.optimize(
            num_iters=NUM_ITERS
        )  # num_iters is for number of boosting iterations

    elif model_name == "LightGBM":
        obj = LightGBM(dict_data["train"], dict_data["val"], dict_data["test"])
        model = obj.optimize(
            num_iters=NUM_ITERS
        )  # num_iters is for number of boosting iterations

    neptune.stop()

    # Get current working directory
    cwd = os.getcwd()
    cwd = cwd[: cwd.find(REPO) + len(REPO)]

    print(
        "Model file save at",
        dump(
            model,
            os.path.join(
                cwd, "src/results/pre-trained_models" + "/" + model_name + ".joblib"
            ),
        ),
    )
