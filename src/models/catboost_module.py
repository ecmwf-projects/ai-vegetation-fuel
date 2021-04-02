import numpy as np
from sklearn.metrics import mean_squared_error as mse
import catboost as cb
from sklearn.preprocessing import PowerTransformer
from catboost import Pool
import joblib

# Parameters
GROW_POLICY = "SymmetricTree"
MAX_DEPTH = 8
MAX_BIN = 145
MIN_DATA_IN_LEAF = 172
L2_LEAF_REG = 7.837674780495601
LEARNING_RATE = 0.1687315081636837
BOOTSTRAP_TYPE = "Bernoulli"
SUBSAMPLE = 0.9979455997654413
EARLY_STOP = 20
VERBOSE_EVAL = 1
RANDOM_SEED = 0

# Save file name for sklearn transform
SCALER_FILENAME = "scaler.save"


class CatBoost:
    """Catboost training module.

    Attributes
    ----------
    train : pandas.core.frame.DataFrame
        Training data.
    val : pandas.core.frame.DataFrame
        Validation data.
    test : pandas.core.frame.DataFrame
        Testing data.
    transform : str
        If any data transform is to be used ,can be 'box_cox' for box-cox transform

    Note
    -----
    When transform is used it is applied only on Fuelload (target variable), not on the training features

    Methods
    -------
    dataloader()
        Preprocesses the data
    fitCat()
        Consists of training module and dictonary of parameters used for the model
    infer()
        Generates metric results from the trained model
    """

    def __init__(self, train, val, test, transform="box_cox"):
        self.train = train
        self.test = test
        self.val = val
        self.transform = transform
        self.model = None
        self.param = {
            "grow_policy": GROW_POLICY,
            "max_depth": MAX_DEPTH,
            "max_bin": MAX_BIN,
            "min_data_in_leaf": MIN_DATA_IN_LEAF,
            "l2_leaf_reg": L2_LEAF_REG,
            "learning_rate": LEARNING_RATE,
            "bootstrap_type": BOOTSTRAP_TYPE,
            "subsample": SUBSAMPLE,
            "verbose": VERBOSE_EVAL,
            "early_stopping_rounds": EARLY_STOP,
            "random_seed": RANDOM_SEED,
        }

    def dataloader(self):

        cols_drop = [
            "actual_load",
        ]
        X_train = self.train.drop(columns=cols_drop)
        y_train = self.train.actual_load
        X_test = self.test.drop(columns=cols_drop)
        y_test = self.test.actual_load
        X_val = self.val.drop(columns=cols_drop)
        y_val = self.val.actual_load

        if self.transform is not None:
            scaler = PowerTransformer(method="box-cox")
            y_train = scaler.fit_transform(
                np.array(self.train.actual_load).reshape(-1, 1)
            )
            y_train = y_train.ravel()

            y_val = scaler.transform(np.array(self.val.actual_load).reshape(-1, 1))
            y_val = y_val.ravel()

            # Saving sklearn transformation file to be further used for inverse transformation in test.py
            scaler_filename = SCALER_FILENAME
            joblib.dump(scaler, scaler_filename)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def objective(self, num_iters):
        (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        ) = self.dataloader()

        model, log = self.fitCat(
            (self.X_train, self.y_train), (self.X_val, self.y_val), iters=num_iters
        )

    def fitCat(self, train, val, iters):
        X_train, y_train = train
        X_val, y_val = val
        params = self.param

        # Training
        train_pool = Pool(X_train, y_train, cat_features=["climatic_region", "month"])
        val_pool = Pool(X_val, y_val, cat_features=["climatic_region", "month"])

        cbr = cb.CatBoostRegressor(**params, allow_writing_files=True)
        self.model = cbr.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
        )

        # Predictions
        pred_val = self.model.predict(val_pool)
        score_val = mse(pred_val, y_val, squared=False)
        print("RMSE  :", score_val)
        log = {
            "train_RMSE": self.model.best_score_["learn"]["RMSE"],
            "val_RMSE": self.model.best_score_["validation"]["RMSE"],
        }

        return self.model, log

    def optimize(self, num_iters):
        self.objective(num_iters)
        self.infer()
        return self.model

    # Metric Results
    def infer(self):

        train_pred = self.model.predict((self.X_train))
        val_pred = self.model.predict((self.X_val))
        test_pred = self.model.predict((self.X_test))
        print("-----------------------------------------------------------------")
        print("Training results", "\n")
        if self.transform is not None:
            scaler = PowerTransformer(method="box-cox")
            scaler.fit(np.array(self.train.actual_load).reshape(-1, 1))
            inv_train_pred = scaler.inverse_transform(
                np.array(train_pred).reshape(-1, 1)
            )
            inv_val_pred = scaler.inverse_transform(np.array(val_pred).reshape(-1, 1))
            inv_test_pred = scaler.inverse_transform(np.array(test_pred).reshape(-1, 1))
            print(
                "Training error: ",
                mse(self.train.actual_load, inv_train_pred, squared=False),
            )
            print(
                "Validation error: ",
                mse(self.val.actual_load, inv_val_pred, squared=False),
            )
            print("Test error: ", mse(self.y_test, inv_test_pred, squared=False))
            print(
                "Note : The error printed above is calculated after the inverse transform of box-cox"
            )

        else:
            print("Training error: ", mse(self.y_train, train_pred, squared=False))
            print("Validation error: ", mse(self.y_val, val_pred, squared=False))
            print("Test error: ", mse(self.y_test, test_pred, squared=False))
