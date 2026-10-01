import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config as cfg
from feature_selection import filter_features, lgbm_feature_selection
from plots import plot_classifier_summary, print_header
from preprocessing import clean_data, engineer_features, save_artifacts, split_features_target

SEED = cfg.RANDOM_STATE


def inv_sq(distances):
    """Inverse-square distance weights for KNN."""
    return 1 / (distances ** 2 + 1e-10)


def timed(func, *args, **kwargs):
    """Call `func` and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


KNN_K_FEATURES = 20

RF_N_ESTIMATORS = [50, 100, 200]
RF_MAX_DEPTHS = [10, 20, 30]

SVM_C_VALUES = [0.1, 1, 10]
SVM_GAMMA_VALUES = ["scale", 0.1, 0.01]
SVM_C, SVM_GAMMA = 10, "scale"  # final SVM; the sweeps above are informational only

DT_PARAM_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [10, 15, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

KNN_K_VALUES = [3, 5, 7, 9, 11]
KNN_WEIGHTS = {"uniform": "uniform", "distance": "distance", "inv_sq": inv_sq}


@dataclass
class ModelResult:
    name: str
    model: object
    accuracy: float
    train_time: float
    test_time: float

    def metrics(self):
        return {"accuracy": self.accuracy, "train_time": self.train_time, "test_time": self.test_time}


@dataclass
class Splits:
    """Train/test data prepared for the two model families."""
    y_train: pd.Series
    y_test: pd.Series
    # RF / SVM / Decision Tree: LightGBM-selected features, scaled
    X_train: np.ndarray
    X_test: np.ndarray
    scaler: StandardScaler
    selected_features: list
    # KNN: further reduced with SelectKBest, then scaled
    X_knn_train: np.ndarray
    X_knn_test: np.ndarray
    knn_scaler: StandardScaler
    knn_selector: SelectKBest
    knn_features: list


#Data 

def load_and_prepare(path=cfg.CLS_DATA_PATH):
    """Read the CSV, engineer features, impute missing values and filter features."""
    df = pd.read_csv(path)
    df = engineer_features(df)
    df, medians, modes = clean_data(df)
    df = filter_features(df, target_col=cfg.CLS_TARGET)
    return df, medians, modes


def prepare_splits(df):
    X, y = split_features_target(df, cfg.CLS_TARGET)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=SEED, stratify=y
    )

    selected = lgbm_feature_selection(X_train, y_train, task="classification")
    print(f"LGBM Selected Features (RF/SVM/DT): {selected}")
    X_train_sel, X_test_sel = X_train[selected], X_test[selected]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    k = min(KNN_K_FEATURES, X_train_sel.shape[1])
    knn_selector = SelectKBest(score_func=f_classif, k=k).fit(X_train_sel, y_train)
    knn_features = X_train_sel.columns[knn_selector.get_support()].tolist()
    print(f"KNN Selected Features: {knn_features}")

    knn_scaler = StandardScaler()
    X_knn_train = knn_scaler.fit_transform(X_train_sel[knn_features])
    X_knn_test = knn_scaler.transform(X_test_sel[knn_features])

    return Splits(y_train, y_test, X_train_scaled, X_test_scaled, scaler, selected,
                  X_knn_train, X_knn_test, knn_scaler, knn_selector, knn_features)


#Training helpers 

def _candidates(values):
    return {str(v): v for v in values}
 
"""Score one hyper-parameter over `candidates` ({label: value}).
   Returns (best_label, best_value). Ties go to the first candidate.
"""
def sweep(title, candidates, make_model, X_train, y_train, X_test, y_test):
   
    print(f"\n{title}")
    accuracies = {}
    for label, value in candidates.items():
        model = make_model(value)
        model.fit(X_train, y_train)
        accuracies[label] = accuracy_score(y_test, model.predict(X_test))
        print(f"    {label:>10} → Accuracy = {accuracies[label]:.4f}")
    best_label = max(accuracies, key=accuracies.get)
    print(f"  Best = {best_label}")
    return best_label, candidates[best_label]


def _score(name, model, train_time, X_test, y_test):
    preds, test_time = timed(model.predict, X_test)
    result = ModelResult(name, model, accuracy_score(y_test, preds), train_time, test_time)
    print(f"{name}: accuracy {result.accuracy * 100:.2f}% | "
          f"train {train_time:.4f}s | test {test_time:.4f}s")
    return result


def _fit_and_score(name, model, X_train, y_train, X_test, y_test):
    _, train_time = timed(model.fit, X_train, y_train)
    return _score(name, model, train_time, X_test, y_test)


# Models 

def _random_forest(n_estimators, max_depth):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  random_state=SEED, n_jobs=-1)


def train_random_forest(s):
    print_header("Training Random Forest")
    X_res, y_res = SMOTEENN(random_state=SEED).fit_resample(s.X_train, s.y_train)

    _, best_n = sweep("[RF Tuning] n_estimators (max_depth=None)",
                      _candidates(RF_N_ESTIMATORS), lambda n: _random_forest(n, None),
                      X_res, y_res, s.X_test, s.y_test)
    _, best_depth = sweep(f"[RF Tuning] max_depth (n_estimators={best_n})",
                          _candidates(RF_MAX_DEPTHS), lambda d: _random_forest(best_n, d),
                          X_res, y_res, s.X_test, s.y_test)

    print(f"\nTraining final RF: n_estimators={best_n}, max_depth={best_depth}")
    return _fit_and_score("Random Forest", _random_forest(best_n, best_depth),
                          X_res, y_res, s.X_test, s.y_test)


def _svm(c, gamma):
    return SVC(kernel="rbf", C=c, gamma=gamma, class_weight="balanced", random_state=SEED)


def train_svm(s):
    print_header("Training SVM")
    sweep("Effect of changing C (gamma='scale')", _candidates(SVM_C_VALUES),
          lambda c: _svm(c, "scale"), s.X_train, s.y_train, s.X_test, s.y_test)
    sweep(f"Effect of changing gamma (C={SVM_C})", _candidates(SVM_GAMMA_VALUES),
          lambda g: _svm(SVM_C, g), s.X_train, s.y_train, s.X_test, s.y_test)

    return _fit_and_score("SVM", _svm(SVM_C, SVM_GAMMA),
                          s.X_train, s.y_train, s.X_test, s.y_test)


def train_decision_tree(s):
    print_header("Training Decision Tree (GridSearchCV)")
    grid = GridSearchCV(DecisionTreeClassifier(random_state=SEED, class_weight="balanced"),
                        DT_PARAM_GRID, cv=5, scoring="accuracy", n_jobs=-1)
    _, train_time = timed(grid.fit, s.X_train, s.y_train)  # includes the full grid search
    print(f"Decision Tree Best Params: {grid.best_params_}")
    return _score("Decision Tree", grid.best_estimator_, train_time, s.X_test, s.y_test)


def _knn(k, weights):
    return KNeighborsClassifier(n_neighbors=k, weights=weights, metric="minkowski", n_jobs=-1)


def train_knn(s):
    print_header("Training KNN")
    _, best_k = sweep("[KNN Tuning] n_neighbors", _candidates(KNN_K_VALUES),
                      lambda k: _knn(k, "uniform"),
                      s.X_knn_train, s.y_train, s.X_knn_test, s.y_test)
    best_weight_label, best_weight = sweep(f"[KNN Tuning] weights (k={best_k})", KNN_WEIGHTS,
                                           lambda w: _knn(best_k, w),
                                           s.X_knn_train, s.y_train, s.X_knn_test, s.y_test)

    result = _fit_and_score("KNN", _knn(best_k, best_weight),
                            s.X_knn_train, s.y_train, s.X_knn_test, s.y_test)
    return result, {"best_k": best_k, "best_w_label": best_weight_label}


# Main

def main():
    warnings.filterwarnings("ignore")
    print_header("Classification Pipeline")

    df, train_medians, train_modes = load_and_prepare()
    splits = prepare_splits(df)

    rf = train_random_forest(splits)
    svm = train_svm(splits)
    dt = train_decision_tree(splits)
    knn, knn_params = train_knn(splits)

    plot_classifier_summary({r.name: r.metrics() for r in (rf, svm, dt, knn)})

    save_artifacts({
        "cls_selected_features": splits.selected_features,
        "cls_scaler": splits.scaler,
        "cls_train_medians": train_medians,
        "cls_train_modes": train_modes,
        "rf_model": rf.model,
        "svm_model": svm.model,
        "dt_model": dt.model,
        "knn_selector": splits.knn_selector,
        "knn_selected_features": splits.knn_features,
        "knn_scaler": splits.knn_scaler,
        "knn_best_params": knn_params,
        "knn_model": knn.model,
    }, cfg.CLS_SAVE_DIR)


if __name__ == "__main__":
    import classification
    classification.main()
