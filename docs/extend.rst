Adding New Features
====================

-  Make sure the new dataset to be added is a single file in ``.nc`` format, containing data from 2010-16 and in 0.25x0.25 grid cell resolution.
-  Match the features of the new dataset with the existing features. This can be done by going through ``src/EDA/EDA_pre-processed_data.ipynb``.
-  Add the feature path as a variable to ``src/utils/data_paths.py``.
   Further the path variable is needed to be added to either the time
   dependant or independant list (depending on which category it belongs
   to) present inside ``export_feature_paths()``.
-  The model will now also be trained on the added feature while running src/train.py!

