Pre-processing
===============

Raw data should first be processed using notebooks in :code:`notebooks/preprocess/*`.
Entry point for the pre-processing script for the ML pipeline is :code:`src/pre-processing.py`.

Args description:
   *``--data_path``:  Path to the data files

-  Input: Enter the root directory of the xarray data files as the script argument.
   All data files produced are stored in this directory.

   -  ``src/utils/data_paths.py`` - defines the files paths for the
      features used in training and also the paths of the
      ``fuel_load.nc`` which will be
      created.

-  Output:

   -  Creates ``fuel_load.nc`` file for Fuel Load Data (Burned Area \*
      Above Ground Biomass).
   -  Saves the following files for the Tropics & Mid-Latitudes regions
      respectively, where {type} is 'tropics' or 'midlats'.
   -  Save Directory ``root_path/{type}`` ::

      -  {type}_train.csv
      -  {type}_val.csv
      -  {type}_test.csv
   -  Save Directory ``root_path/infer_{type}`` ::

      -  {type}*infers*\ July.csv
      -  {type}*infers*\ Aug.csv
      -  {type}*infers*\ Sept.csv
      -  {type}*infers*\ Oct.csv
      -  {type}*infers*\ Nov.csv
      -  {type}*infers*\ Dec.csv

   Where root\_path is the root save path provided for ``pre-processing.py``
