LightGBM inference
==================

This notebooks demonstrates generating inferences from a pretrained
LightGBM model. This notebook utilizes the ``deepfuel-ML/src/test.py``
script for generating inferences. The script does everything from
calculating error values to plotting data for visual inference.

.. code:: ipython3

    import os
    import pandas as pd
    import numpy as np
    from joblib import dump, load
    import sys
    import os
    from IPython.display import Image, display

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

    !python '../src/test.py'  --model_name 'LightGBM' --model_path '../src/pre-trained_models/LightGBM.joblib' --data_path '../data/infer_tropics'  --results_path '../data/tropics/results'


.. parsed-literal::

    MAPE July : 358.2370533961142
    MAPE Aug : 4068.041474465497
    MAPE Sept : 342.60497263841376
    MAPE Oct : 407.02247341732897
    MAPE Nov : 553.79772310129
    MAPE Dec : 433.6634326468742
    Actual FL plot successfully generated! File saved to  ../data/tropics/results/tropics_Nov_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/tropics/results/tropics_Nov_predicted.html
    Actual FL plot successfully generated! File saved to  ../data/tropics/results/tropics_Aug_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/tropics/results/tropics_Aug_predicted.html
    Actual FL plot successfully generated! File saved to  ../data/tropics/results/tropics_Dec_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/tropics/results/tropics_Dec_predicted.html
    Actual FL plot successfully generated! File saved to  ../data/tropics/results/tropics_Oct_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/tropics/results/tropics_Oct_predicted.html
    Actual FL plot successfully generated! File saved to  ../data/tropics/results/tropics_July_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/tropics/results/tropics_July_predicted.html
    Actual FL plot successfully generated! File saved to  ../data/tropics/results/tropics_Sept_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/tropics/results/tropics_Sept_predicted.html


Inference CSV
~~~~~~~~~~~~~

``test.py`` generates ``.csv`` files for each month with the following
columns: - ``latitude`` - ``longitude`` - ``actual_load`` - Actual Fuel
Load value - ``predicted_load`` - Predicted Fuel Load value - ``APE`` -
Average Percentage Error between actual and predicted fuel load values

.. code:: ipython3

    df=pd.read_csv('../data/tropics/results/tropics_output_July.csv')
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
          <td>-29.875</td>
          <td>29.125</td>
          <td>1.876688e+08</td>
          <td>6.441964e+08</td>
          <td>243.262403</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-29.875</td>
          <td>29.375</td>
          <td>2.971511e+08</td>
          <td>3.617555e+08</td>
          <td>21.741276</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-29.875</td>
          <td>29.625</td>
          <td>1.518198e+08</td>
          <td>3.590228e+08</td>
          <td>136.479556</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-29.875</td>
          <td>29.875</td>
          <td>3.022351e+08</td>
          <td>3.368480e+08</td>
          <td>11.452295</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-29.875</td>
          <td>30.125</td>
          <td>3.009002e+08</td>
          <td>3.559008e+08</td>
          <td>18.278682</td>
        </tr>
      </tbody>
    </table>
    </div>



Without Ground Truth (``actual_load`` is not present in the test csv)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    !python '../src/test.py'  --model_name 'LightGBM' --model_path '../src/pre-trained_models/LightGBM.joblib' --data_path '../data/infer_tropics'  --results_path '../data/tropics/results'


.. parsed-literal::

    MAPE July : 358.2370533961142
    MAPE Aug : 4068.041474465497
    MAPE Sept : 342.60497263841376
    MAPE Oct : 407.02247341732897
    MAPE Nov : 553.79772310129
    MAPE Dec : 433.6634326468742
    Actual FL plot successfully generated! File saved to  ../data/tropics/results/tropics_Nov_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/tropics/results/tropics_Nov_predicted.html
    Actual FL plot successfully generated! File saved to  ../data/tropics/results/tropics_Aug_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/tropics/results/tropics_Aug_predicted.html
    Actual FL plot successfully generated! File saved to  ../data/tropics/results/tropics_Dec_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/tropics/results/tropics_Dec_predicted.html
    Actual FL plot successfully generated! File saved to  ../data/tropics/results/tropics_Oct_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/tropics/results/tropics_Oct_predicted.html
    Actual FL plot successfully generated! File saved to  ../data/tropics/results/tropics_July_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/tropics/results/tropics_July_predicted.html
    Actual FL plot successfully generated! File saved to  ../data/tropics/results/tropics_Sept_actual.html
    Predicted FL plot successfully generated! File saved to  ../data/tropics/results/tropics_Sept_predicted.html


Inference CSV
~~~~~~~~~~~~~

.. code:: ipython3

    df=pd.read_csv('../data/tropics/results/tropics_output_July.csv')
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
          <td>-29.875</td>
          <td>29.125</td>
          <td>1.876688e+08</td>
          <td>6.441964e+08</td>
          <td>243.262403</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-29.875</td>
          <td>29.375</td>
          <td>2.971511e+08</td>
          <td>3.617555e+08</td>
          <td>21.741276</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-29.875</td>
          <td>29.625</td>
          <td>1.518198e+08</td>
          <td>3.590228e+08</td>
          <td>136.479556</td>
        </tr>
        <tr>
          <th>3</th>
          <td>-29.875</td>
          <td>29.875</td>
          <td>3.022351e+08</td>
          <td>3.368480e+08</td>
          <td>11.452295</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-29.875</td>
          <td>30.125</td>
          <td>3.009002e+08</td>
          <td>3.559008e+08</td>
          <td>18.278682</td>
        </tr>
      </tbody>
    </table>
    </div>



Visualizing the plots generated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The plots are stored as html files that can be zoomed in upto the
resolution of the data to view the predicted and actual values.
