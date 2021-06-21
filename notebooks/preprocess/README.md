# Notebooks for preprocessing completely raw data

These notebooks demonstrate (pre?)preprocessing for data obtained from heterogenous sources
various formats. We also have a notebook demonstrating the steps we undertake to convert
data from `grib` or `grib2` file format to the `netcdf` format used in the project. 

## Directory structure

The raw data is expected to be in a directory `raw/` at the same level as
these notebooks. The notebooks output the preprocessed data into a `preprocess/` 
directory that is also located at the same level as the notebooks. In our typical work-
flow, we use symbolic links to achieve this, because often times such large datasets of 
raw files are present on data disks mounted elsewhere. 

For example, if the working directory is `~/ai-vegetation-fuel/`, the raw data are in `/data1/raw_data/`, and we wish to store the processed
data in `/data1/processed_data/`, we would create symbolic links:

```bash
ln -s /data1/raw_data ./notebooks/preprocess/raw
ln -s /data1/processed_data ./notebooks/preprocess/preprocess
```

This should give you a directory structure like this (`ls -l ./notebooks/preprocess/`):

```bash
.
├── preprocess -> /data1/processed_data
├── preprocess_agb.ipynb
├── preprocess_burned_area.ipynb
├── preprocess_climate_regions.ipynb
├── preprocess_fire_anomalies.ipynb
├── preprocess_leaf_area_index.ipynb
├── preprocess_slopes.ipynb
├── preprocess_spi_gpcc.ipynb
├── preprocess_weather_anomalies.ipynb
├── raw -> /data1/raw_data
└── README.md
```

You can then create a symlink in the root of the repository with the name `data/` to 
where you store the preprocessed data.

```bash
ln -s /data1/processed_data ./data
```

to get the following tree:

```bash
.
├── data -> /data1/preprocessed_data
├── dev-requirements.txt
├── docker
├── docs
│   └── _static
├── LICENSE
├── notebooks
│   └── preprocess
│       ├── preprocess -> /data1/preprocessed_data
│       └── raw -> /data1/raw_data
├── README.md
├── requirements.txt
└── src
    ├── models
    ├── pre-trained_models
    └── utils
```

The Above Ground Biomass (AGB) dataset should be preprocessed first, since all other datasets use it as a baseline for resolution and dimensions. 
