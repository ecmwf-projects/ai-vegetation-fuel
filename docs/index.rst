.. smos-fuel documentation master file, created by
   sphinx-quickstart on Wed Mar 24 22:41:35 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

smos-fuel: Predicting Fuel Load with Machine Learning on Climate Data
======================================================================

We use SMOS derived products along with other climatic and topographical data
to predict Fuel Load, which embodies the content of burnable vegetation in an area.

Two models are developed, for the Mid-Latitudes and the Tropics using Gradient Boosted
Decision Tree style Machine Learning methods. We can see below a visual comparison of the
predictions made by the model and corresponding ground truth.

-  CatBoost for Mid-Latitudes

   .. image:: _static/mid-lat.png

-  LightGBM for Tropics

   .. image:: _static/tropic.png

We recommend you go through the Getting Started section to first set up your development environment.
Once you have the input data (as ``.nc`` NetCDF files) and the dependencies installed,
you can either refer to the example notebooks
in ``demo-notebooks/`` or go through the further notes on preprocessing, training and testing. More details
can be found in the Module API docs in the index below.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   intro
   data
   preprocessing
   training
   testing
   extend


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
