Getting Started
===============

Installation
-------------

The python environment for the repository can be created using
**either** ``conda`` or ``virtualenv``, by running from the root of the
repo:

Using conda
^^^^^^^^^^^

.. code:: bash

    conda create --name=ml-fuel python=3.8
    conda activate ml-fuel

Using virtualenv
^^^^^^^^^^^^^^^^

.. code:: bash

    python3 -m venv env
    source env/bin/activate

Install dependencies
^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    pip install -U pip
    pip install -r requirements.txt

This includes all the packages required for running the code in the
repository.

Pre-trained models
------------------

Pre-trained models are available:
 - ``LightGBM.joblib`` at ``src/results/pre-trained_models/LightGBM.joblib``
 - ``CatBoost.joblib`` at ``src/results/pre-trained_models/CatBoost.joblib``

Demo Notebooks
---------------

Notebooks for training and inference:
 - ``LightGBM_training.ipynb`` at ``demo-notebooks/LightGBM_training.ipynb``
 - ``LightGBM_inference.ipynb`` at  ``demo-notebooks/LightGBM_inference.ipynb``
 - ``CatBoost_training.ipynb`` at ``demo-notebooks/CatBoost_training.ipynb``
 - ``CatBoost_inference.ipynb`` at  ``demo-notebooks/CatBoost_inference.ipynb``
