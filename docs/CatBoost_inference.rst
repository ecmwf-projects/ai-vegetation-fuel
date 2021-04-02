CatBoost inference
==================

This notebooks demonstrates generating inferences from a pretrained
CatBoost model. This notebook utilizes the ``deepfuel-ML/src/test.py``
script for generating inferences. The script does everything from
calculating error values to plotting data for visual inference.

.. code:: ipython3

    import os
    import pandas as pd
    import numpy as np
    from joblib import dump, load
    from IPython.display import Image, display, HTML

Using ``test.py``
~~~~~~~~~~~~~~~~~

Below is the description of its arguements: - ``--model_name``: Name of
the model to be trained (“CatBoost” or “LightGBM”). - ``--model_path``:
Path to the pre-trained model. - ``--data_path``: Valid data directory
where all the test .csv files are stored. - ``--results_path``:
Directory where the result inference .csv files and .png visualizations
are going to be stored.

With Ground Truth (``actual_load`` is present in the test csv)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    !python '../src/test.py'  --model_name 'CatBoost' --model_path '../src/pre-trained_models/CatBoost.joblib' --data_path '../data/infer_midlats'  --results_path '../data/midlats/results'


.. parsed-literal::

    MAPE July : 380.44795759521344
    MAPE Aug : 283.7487728040964
    MAPE Sept : 203.97476414457114
    MAPE Oct : 117.19251658203949
    MAPE Nov : 105.94428641567805
    MAPE Dec : 99.29645055040669
    Actual FL plot successfully generated! File saved to  ../data/midlats/results/midlats_Nov_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/midlats/results/midlats_Nov_predicted.html
    Actual FL plot successfully generated! File saved to  ../data/midlats/results/midlats_July_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/midlats/results/midlats_July_predicted.html
    Actual FL plot successfully generated! File saved to  ../data/midlats/results/midlats_Dec_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/midlats/results/midlats_Dec_predicted.html
    Actual FL plot successfully generated! File saved to  ../data/midlats/results/midlats_Aug_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/midlats/results/midlats_Aug_predicted.html
    Actual FL plot successfully generated! File saved to  ../data/midlats/results/midlats_Oct_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/midlats/results/midlats_Oct_predicted.html
    Actual FL plot successfully generated! File saved to  ../data/midlats/results/midlats_Sept_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/midlats/results/midlats_Sept_predicted.html


Inference CSV
~~~~~~~~~~~~~

``test.py`` generates ``.csv`` files for each month with the following
columns: - ``latitude`` - ``longitude`` - ``actual_load`` - Actual Fuel
Load value - ``predicted_load`` - Predicted Fuel Load value - ``APE`` -
Average Percentage Error between actual and predicted fuel load values

.. code:: ipython3

    df=pd.read_csv('../data/midlats/results/midlats_output_July.csv')
    df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>lat</th>
          <th>lon</th>
          <th>actual_load</th>
          <th>predicted_load</th>
          <th>APE</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>-35.125</td>
          <td>-69.375</td>
          <td>9.188477e+07</td>
          <td>8.817028e+07</td>
          <td>4.042547</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-31.625</td>
          <td>27.875</td>
          <td>7.486465e+07</td>
          <td>5.130763e+08</td>
          <td>585.338529</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-31.375</td>
          <td>28.375</td>
          <td>6.728101e+07</td>
          <td>4.373534e+08</td>
          <td>550.039875</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-31.125</td>
          <td>28.625</td>
          <td>9.200570e+07</td>
          <td>4.966761e+08</td>
          <td>439.831873</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-31.125</td>
          <td>29.625</td>
          <td>1.413486e+08</td>
          <td>4.879350e+08</td>
          <td>245.199817</td>
        </tr>
      </tbody>
    </table>
    </div>



Without Ground Truth (``actual_load`` is not present in the test csv)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    !python '../src/test.py'  --model_name 'CatBoost' --model_path '../src/pre-trained_models/CatBoost.joblib' --data_path '../data/infer_midlats'  --results_path '../data/midlats/results'


.. parsed-literal::

    MAPE July : 380.44795759521344
    MAPE Aug : 283.7487728040964
    MAPE Sept : 203.97476414457114
    MAPE Oct : 117.19251658203949
    MAPE Nov : 105.94428641567805
    MAPE Dec : 99.29645055040669
    Actual FL plot successfully generated! File saved to  ../data/midlats/results/midlats_Nov_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/midlats/results/midlats_Nov_predicted.html
    Actual FL plot successfully generated! File saved to  ../data/midlats/results/midlats_July_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/midlats/results/midlats_July_predicted.html
    Actual FL plot successfully generated! File saved to  ../data/midlats/results/midlats_Dec_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/midlats/results/midlats_Dec_predicted.html
    Actual FL plot successfully generated! File saved to  ../data/midlats/results/midlats_Aug_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/midlats/results/midlats_Aug_predicted.html
    Actual FL plot successfully generated! File saved to  ../data/midlats/results/midlats_Oct_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/midlats/results/midlats_Oct_predicted.html
    Actual FL plot successfully generated! File saved to  ../data/midlats/results/midlats_Sept_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/midlats/results/midlats_Sept_predicted.html


Inference CSV
~~~~~~~~~~~~~

.. code:: ipython3

    df=pd.read_csv('../data/midlats/results/midlats_output_July.csv')
    df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>lat</th>
          <th>lon</th>
          <th>actual_load</th>
          <th>predicted_load</th>
          <th>APE</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>-35.125</td>
          <td>-69.375</td>
          <td>9.188477e+07</td>
          <td>8.817028e+07</td>
          <td>4.042547</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-31.625</td>
          <td>27.875</td>
          <td>7.486465e+07</td>
          <td>5.130763e+08</td>
          <td>585.338529</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-31.375</td>
          <td>28.375</td>
          <td>6.728101e+07</td>
          <td>4.373534e+08</td>
          <td>550.039875</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-31.125</td>
          <td>28.625</td>
          <td>9.200570e+07</td>
          <td>4.966761e+08</td>
          <td>439.831873</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-31.125</td>
          <td>29.625</td>
          <td>1.413486e+08</td>
          <td>4.879350e+08</td>
          <td>245.199817</td>
        </tr>
      </tbody>
    </table>
    </div>



Visualizing the plots generated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The plots are stored as html files that can be zoomed in upto the
resolution of the data to view the predicted and actual values
