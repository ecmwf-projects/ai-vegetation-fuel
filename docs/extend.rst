Adding New Features
====================

-  Make sure the dataset is in ``.nc`` format.
-  Match resolution of the feature with the existing features. This can
   be done by going through ``src/EDA/EDA_pre-processed_data.ipynb``.
-  Add the feature path as a variable to ``src/utils/data_paths.py``.
   Further the path variable is needed to be added to either the time
   dependant or independant list (depending on which category it belongs
   to) present inside ``export_feature_paths()``.
-  The model will now also be trained on the added feature!

.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
