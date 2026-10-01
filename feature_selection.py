import re
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import config as cfg
from plots import plot_top_features, print_header

#Drop near-constant columns, then one column of every highly correlated pair
def filter_features(df, target_col, variance_threshold=0.995, correlation_threshold=0.85):
    print_header("Starting Feature Filtering (Variance & Correlation)")
    df = df.copy()

    # 1. Near-zero variance: the most frequent value covers >= threshold of rows.
    low_variance = [
        col for col in df.columns
        if col != target_col
        and df[col].value_counts(normalize=True).iloc[0] >= variance_threshold
    ]
    if low_variance:
        df = df.drop(columns=low_variance)
        print(f"Removed {len(low_variance)} Zero/Low Variance Features: {low_variance}")

    # 2. High correlation: keep the first column of each pair, drop the later one.
    corr = df.select_dtypes(include=[np.number]).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [
        col for col in upper.columns
        if col != target_col and (upper[col] > correlation_threshold).any()
    ]
    if high_corr:
        df = df.drop(columns=high_corr)
        print(f"Removed {len(high_corr)} Highly Correlated Features: {high_corr}")

    return df

#Rank features by LightGBM gain importance and return the `top_n` names
def lgbm_feature_selection(X, y, task="regression", top_n=10):
    
    print_header(f"Starting LightGBM Feature Selection ({task.capitalize()})")

    # LightGBM rejects some special characters, so fit on sanitised names and map back.
    sanitise = lambda name: re.sub(r"[^A-Za-z0-9_]+", "", name)
    original_names = {sanitise(col): col for col in X.columns}
    X_safe = X.rename(columns=sanitise)

    params = dict(n_estimators=250, learning_rate=0.05, importance_type="gain",
                  random_state=cfg.RANDOM_STATE, n_jobs=-1, verbose=-1)
    if task == "classification":
        model = lgb.LGBMClassifier(**params)
        y_ready = LabelEncoder().fit_transform(y)
    else:
        model = lgb.LGBMRegressor(**params)
        y_ready = y
    model.fit(X_safe, y_ready)

    importances = model.feature_importances_
    ranking = pd.DataFrame({
        "Feature": [original_names[c] for c in X_safe.columns],
        "Relative_Importance_%": importances / importances.sum() * 100,
        "Importance_Gain": importances,
    }).sort_values(by="Importance_Gain", ascending=False)

    top = ranking.head(top_n)[["Feature", "Relative_Importance_%"]]
    print(f"\nTop {top_n} features by LightGBM gain:")
    print(top.to_string(index=False))
    plot_top_features(top, f"Top {top_n} Features by LightGBM Gain")

    return top["Feature"].tolist()
