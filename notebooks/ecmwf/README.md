# Notebooks for pre-processing and modelling

Notebooks in this folder contain all the steps for data exploration, pre-processing, modelling and explainability using the H2O.ai framework. 

#### Install dependencies

```bash
conda install cartopy
conda install -c h2oai h2o
```

## Notebooks

### 1. Data preparation

This notebook takes the raw/downloaded information and pre-processes it into a data frame. The data is then split into a train and test set using a stratified sampling strategy to make sure both datasets have the same proportion of biomes.

### 2. Exploratory data analysis

This notebook explores the data assembled in notebook `data_preparation.ipynb`. It looks at the probability distributions of outcome and predictors and identifies possible data transformations as well as correlations and redundandies amongst variables.

### 3. Model benchmark tests

This notebook uses the H2O.ai AutoML framework to benchmark various possible data transformations for outcome and predictors. It also compares model results in case all versus non-redundant features are used. The final result is a pre-processed dataset that will be used for the final modelling step in `model_definition.ipynb`.

### 4. Model definition and evaluation

This notebook uses the H2O.ai AutoML framework to model transformed outcome and predictors. It visualises averaged results over a map and uses the H2O.ai explainability module to identify model limitations and possible future avenues for improvements.


## Info

These notebooks were developed by the Wildfire Danger Forecasting team at the European Centre for Medium-range Weather Forecast for the ESA-SMOS-2020 project. For any queries, please contact ECMWF support portal: https://confluence.ecmwf.int/site/support.
