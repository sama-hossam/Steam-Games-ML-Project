import warnings
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

import config as cfg
from feature_selection import filter_features, lgbm_feature_selection
from plots import print_header
from preprocessing import clean_data, engineer_features, save_artifacts, split_features_target

N_TRIALS = 15
CV_FOLDS = 5
LOG_PREDICTION_CAP = 20  # clip log-scale predictions before expm1 to avoid overflow

XGB_FIXED_PARAMS = {"objective": "reg:squarederror", "random_state": cfg.RANDOM_STATE, "n_jobs": -1}
LGB_FIXED_PARAMS = {"random_state": cfg.RANDOM_STATE, "n_jobs": -1, "verbose": -1}


# Data 

def load_and_prepare(path=cfg.REG_DATA_PATH):
    """Read the CSV, engineer features, filter them and impute missing values."""
    df = pd.read_csv(path)
    df = engineer_features(df)
    df = filter_features(df, target_col=cfg.REG_TARGET)
    return clean_data(df)  # (df, train_medians, train_modes)


# Hyper-parameter tuning

def _suggest_xgb_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def _suggest_lgb_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def tune_model(name, estimator_cls, suggest_params, fixed_params, X, y, n_trials=N_TRIALS):
    """Optuna search maximising 5-fold CV R²; returns the best parameter dict."""
    print_header(f"Tuning {name}...")
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)

    def objective(trial):
        model = estimator_cls(**suggest_params(trial), **fixed_params)
        return cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1).mean()

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = {**study.best_params, **fixed_params}
    print(f"Best {name} Params:\n{best_params}")
    return best_params


# Evaluation 

def evaluate(model, X_test, y_test):
    """R² on the log scale; RMSE / MAE on the original (expm1) scale."""
    pred_log = model.predict(X_test)
    y_true = np.expm1(y_test)
    y_pred = np.expm1(np.clip(pred_log, a_min=None, a_max=LOG_PREDICTION_CAP))
    return {
        "r2": r2_score(y_test, pred_log),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
    }


# Main
#run all pipeline
def main():
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Data
    df, train_medians, train_modes = load_and_prepare()
    X, y = split_features_target(df, cfg.REG_TARGET)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE
    )

    # Feature selection + scaling
    selected_features = lgbm_feature_selection(X_train, y_train, task="regression")
    print(f"Selected Features: {selected_features}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[selected_features])
    X_test_scaled = scaler.transform(X_test[selected_features])

    # Base models
    xgb_params = tune_model("XGBoost", xgb.XGBRegressor, _suggest_xgb_params,
                            XGB_FIXED_PARAMS, X_train_scaled, y_train)
    lgb_params = tune_model("LightGBM", lgb.LGBMRegressor, _suggest_lgb_params,
                            LGB_FIXED_PARAMS, X_train_scaled, y_train)

    xgb_model = xgb.XGBRegressor(**xgb_params)
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    xgb_model.fit(X_train_scaled, y_train)
    lgb_model.fit(X_train_scaled, y_train)
    print("R² of XGBoost model:", r2_score(y_test, xgb_model.predict(X_test_scaled)))
    print("R² of LightGBM model:", r2_score(y_test, lgb_model.predict(X_test_scaled)))

    # Stacking
    print_header("Training Stacking Model...")
    stacking_model = StackingRegressor(
        estimators=[("xgb", xgb_model), ("lgb", lgb_model)],
        final_estimator=Ridge(alpha=1.0),
        cv=CV_FOLDS,
        n_jobs=-1,
    )
    stacking_model.fit(X_train_scaled, y_train)

    metrics = evaluate(stacking_model, X_test_scaled, y_test)
    cv_scores = cross_val_score(
        stacking_model, X, y,
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE),
        scoring="r2", n_jobs=-1,
    )

    print("\nFinal Performance:")
    print(f"R² Score (Test): {metrics['r2']:.4f}")
    print(f"R² CV Mean: {cv_scores.mean():.4f}")
    print(f"R² CV Std : {cv_scores.std():.4f}")
    print(f"RMSE      : {metrics['rmse']:.2f}")
    print(f"MAE       : {metrics['mae']:.2f}")

    # Persist
    save_artifacts({
        "selected_features": selected_features,
        "scaler": scaler,
        "stacking_model": stacking_model,
        "xgb_model": xgb_model,
        "lgb_model": lgb_model,
        "train_medians": train_medians,
        "train_modes": train_modes,
    }, cfg.REG_SAVE_DIR)


if __name__ == "__main__":
    main()
