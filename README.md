# ml-fuel: Predicting Fuel Load for Wildfire Modelling

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/ml-fuel/badge/?version=latest)](https://ml-fuel.readthedocs.io/en/latest/?badge=latest)

### Getting Started

The python environment for the repository can be created using **either** `conda` or `virtualenv`, by running from the root of the repo:

#### Using conda

```bash
conda create --name=ml-fuel python=3.8
conda activate ml-fuel
```

#### Using virtualenv

```bash
python3 -m venv env
source env/bin/activate
```

#### Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

This includes all the packages required for running the code in the repository, with the exclusion of the notebooks in the folder `notebooks/ecmwf` (see `notebooks/ecmwf/README.md` for the additional dependencies to install).

The content of this repository is split into 2 types of experiments:

1. target is the fuel load = burned areas * above ground biomass
2. target is dry matter = burned areas * above ground biomass * combustion coefficients / grid cell areas

## Experiment 1

### Data Description
7 years of global historical data, from 2010 - 2016 will be used for developing the machine learning models. All data used in this project is propietary and NOT meant for public release. Xarray, NumPy and netCDF libraries are used for working with the multi-dimensional geospatial data.
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

## Pre-processing
Raw data should first be processed using notebooks in `notebooks/preprocess/*`.
Entry point for the pre-processing script for the ML pipeline is [src/pre-processing.py](src/pre-processing.py).
```
Args description:
      * `--data_path`:  Path to the data files.
```
 - Input: Enter the root directory of the xarray data files as the script argument. All data files produced are stored in this directory.
     - `src/utils/data_paths.py` - defines the files paths for the features used in training and the paths of `fuel_load.nc` which will be created.
 - Output:
     - Creates `fuel_load.nc` file for Fuel Load Data (Burned Area * Above Ground Biomass).
     - Saves the following files for the Tropics & Mid-Latitudes regions respectively, where {type} is 'tropics' or 'midlats'.
    ```
          Save Directory root_path/{type}
          * {type}_train.csv
          * {type}_val.csv
          * {type}_test.csv
          Save Directory root_path/infer_{type}
          * {type}_infers_July.csv
          * {type}_infers_Aug.csv
          * {type}_infers_Sept.csv
          * {type}_infers_Oct.csv
          * {type}_infers_Nov.csv
          * {type}_infers_Dec.csv

      Where root_path is the root save path provided for pre-processing.py
     ```

### Training

Entry-point for training is [src/train.py](src/train.py)
```
Args description:
      * `--model_name`:  Name of the model to be trained ("CatBoost" or "LightGBM").
      * `--data_path`:  Data directory where all the input (train, val, test) .csv files are stored.
      * `--exp_name`:  Name of the  training experiment used for logging.
```

### Inference

Entry-point for inference is [src/test.py](src/test.py)
```
Args description:
      * `--model_name`:  Name of the model to be trained ("CatBoost" or "LightGBM").
      * `--model_path`:  Path to the pre-trained model.
      * `--data_path`:  Valid data directory where all the test .csv files are stored.
      * `--results_path`:  Directory where the result inference .csv files and .html visualizations are going to be stored.
```

#### Pre-trained models
Pre-trained models are available at:
- [LightGBM.joblib](src/pre-trained_models/LightGBM.joblib)
- [CatBoost.joblib](src/pre-trained_models/CatBoost.joblib)


#### Demo Notebooks
Notebooks for training and inference:
- [LightGBM_training.ipynb](notebooks/LightGBM_training.ipynb)
- [LightGBM_inference.ipynb](notebooks/LightGBM_inference.ipynb)
- [CatBoost_training.ipynb](notebooks/CatBoost_training.ipynb)
- [CatBoost_inference.ipynb](notebooks/CatBoost_inference.ipynb)

### Fuel Load Prediction Visualizations:
- CatBoost for Mid-Latitudes
<img width="1025" alt="midlats-prediction-july16" src="https://user-images.githubusercontent.com/7680686/113362982-4d263500-936d-11eb-922e-5a0609e7a67e.png">

- LightGBM for Tropics
<img width="1025" alt="tropics-prediction-july16" src="https://user-images.githubusercontent.com/7680686/113362967-45ff2700-936d-11eb-93a3-5ad380393f03.png">


### Adding New Features:
- Make sure the new dataset to be added is a single file in `.nc` format, containing data from 2010-16 and in 0.25x0.25 grid cell resolution.
- Match the features of the new dataset with the existing features. This can be done by going through `notebooks/EDA_pre-processed_data.ipynb`.
- Add the feature path as a variable to `src/utils/data_paths.py`. Further the path variable is needed to be added to either the time dependant or independant list (depending on which category it belongs to) present inside `export_feature_paths()`.
- The model will now also be trained on the added feature while running src/train.py!


### Documentation
Documentation is available at: [https://ml-fuel.readthedocs.io/en/latest/index.html](https://ml-fuel.readthedocs.io/en/latest/index.html).


## Experiment 2
We employ an AutoML approach to predict dry matter using the H2O.ai AutoML framework. Please refer to `notebooks/ecmwf/README.md` for a description of this experiment, instructions to install additional dependencies and the notebooks with the steps to perform the experiment.


## Info
This repository was developed by Anurag Saha Roy (@lazyoracle) and Roshni Biswas (@roshni-b) for the ESA-SMOS-2020 project. Contact email: `info@wikilimo.co`. The repository is now maintained by the Wildfire Danger Forecasting team at the European Centre for Medium-range Weather Forecast.
