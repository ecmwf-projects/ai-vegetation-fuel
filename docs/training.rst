Training
=========

Entry-point for train
---------------------
``src/train.py``

::

    Args description:
          * `--model_name`:  Name of the model to be trained ("CatBoost" or "LightGBM").
          * `--data_path`:  Data directory where all the input (train, val, test) .csv files are stored.
          * `--exp_name`:  Name of the  training experiment used for logging.
