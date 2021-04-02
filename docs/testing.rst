Inference
==========

Entry-point for inference is ``src/test.py``

Arguments description:

    * ``--model_name``:  Name of the model to be trained (``CatBoost`` or ``LightGBM``).
    * ``--model_path``:  Path to the pre-trained ``.joblib`` model.
    * ``--data_path``:  Valid data directory where all the test ``.csv`` files are stored.
    * ``--results_path``:  Directory where the result inference ``.csv`` files and ``.html`` visualizations are going to be stored.
