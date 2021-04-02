from sklearn.metrics import mean_squared_error as mse
import lightgbm as lgb
from neptunecontrib.monitoring.lightgbm import neptune_monitor

# Parameters
BOOSTING = "gbdt"
NUM_LEAVES = 689
MAX_DEPTH = 14
MAX_BIN = 233
MIN_DATA_IN_LEAF = 11
MIN_DATA_IN_BIN = 2
MIN_GAIN_TO_SPLIT = 4.859999999999999
LAMDA_L1 = 0.06701950845750886
LAMDA_L2 = 0.09071591942212155
LEARNING_RATE = 0.007307246335001392
BAGGING_FRACTION = 0.51
FEATURE_FRACTION = 0.74
METRIC = "RMSE"
OBJECTIVE = "regression"
EARLY_STOP = 20
VERBOSE_EVAL = 1
RANDOM_STATE = 0


class LightGBM:
    """LightGBM training module.

    Attributes
    ----------
    train : pandas.core.frame.DataFrame
        Training data.
    val : pandas.core.frame.DataFrame
        Validation data.
    test : pandas.core.frame.DataFrame
        Testing data.

    Methods
    -------
    dataloader()
        Preprocesses the data
    fitLGBM()
        Consists of training module and dictonary of parameters used for the model
    infer()
        Generates metric results from the trained model
    """

    def __init__(self, train, val, test):
        self.train = train
        self.test = test
        self.val = val
        self.model = None
        self.param = {
            "boosting": BOOSTING,
            "num_leaves": NUM_LEAVES,
            "max_depth": MAX_DEPTH,
            "max_bin": MAX_BIN,
            "min_data_in_leaf": MIN_DATA_IN_LEAF,
            "min_data_in_bin": MIN_DATA_IN_BIN,
            "min_gain_to_split": MIN_GAIN_TO_SPLIT,
            "lamda_l1": LAMDA_L1,
            "lamda_l2": LAMDA_L2,
            "learning_rate": LEARNING_RATE,
            "bagging_fraction": BAGGING_FRACTION,
            "feature_fraction": FEATURE_FRACTION,
            "metric": METRIC,
            "early_stopping_rounds": EARLY_STOP,
            "objective": OBJECTIVE,
            "verbose": VERBOSE_EVAL,
            "random_state": RANDOM_STATE,
        }

    def dataloader(self):

        cols_drop = ["actual_load"]
        X_train = self.train.drop(columns=cols_drop)
        y_train = self.train.actual_load
        X_test = self.test.drop(columns=cols_drop)
        y_test = self.test.actual_load
        X_val = self.val.drop(columns=cols_drop)
        y_val = self.val.actual_load

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

        model, yPredValid, log = self.fitLGBM(
            (self.X_train, self.y_train), (self.X_val, self.y_val), iters=num_iters
        )

    def fitLGBM(self, train, val, iters):
        X_train, y_train = train
        X_val, y_val = val

        params = self.param

        # Training

        dTrain = lgb.Dataset(X_train, label=y_train)
        dVal = lgb.Dataset(
            X_val,
            label=y_val,
        )

        watchlist = [dTrain, dVal]

        self.model = lgb.train(
            params,
            train_set=dTrain,
            num_boost_round=iters,
            valid_sets=watchlist,
            valid_names=["train", "val"],
            callbacks=[neptune_monitor()],
        )
        pred_val = self.model.predict(X_val, num_iteration=self.model.best_iteration)

        # Predictions
        score_val = mse(pred_val, y_val, squared=False)
        print("RMSE  :", score_val)
        log = {
            "train_RMSE": self.model.best_score["train"]["rmse"],
            "val_RMSE": self.model.best_score["val"]["rmse"],
        }

        return self.model, pred_val, log

    def optimize(self, num_iters):
        self.objective(num_iters)
        self.infer()
        return self.model

    # Metrics Results
    def infer(self):

        train_pred = self.model.predict((self.X_train))
        val_pred = self.model.predict((self.X_val))
        test_pred = self.model.predict((self.X_test))
        print("-----------------------------------------------------------------")
        print("Training results", "\n")
        print("Training error: ", mse(self.y_train, train_pred, squared=False))
        print("Validation error: ", mse(self.y_val, val_pred, squared=False))
        print("Test error: ", mse(self.y_test, test_pred, squared=False))
