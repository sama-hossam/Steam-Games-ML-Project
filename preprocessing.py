import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from plots import corr_heatmap, plot_distributions, print_header


# DATA CLEANING
   
#handeling missing values in numerical and categorical columns
def clean_data(df, medians=None, modes=None):

    df, medians = _impute_numerical(df, medians)
    df, modes = _impute_categorical(df, modes)
    return df, medians, modes


def _impute_numerical(df, medians=None):
    df = df.copy()
    columns = df.select_dtypes(include=[np.number]).columns
    fitting = medians is None
    if fitting:
        medians = {col: df[col].median() for col in columns}

    filled = []
    for col in columns:
        n_nulls = int(df[col].isnull().sum())
        if n_nulls == 0:
            continue
        value = medians.get(col)
        if value is None:
            value = 0
        df[col] = df[col].fillna(value)
        filled.append((col, n_nulls, round(float(value), 4)))

    source = "Training Median" if fitting else "Saved Training Median"
    _print_fill_report("numerical", source, "Median Used", filled, value_width=12)
    return df, medians


def _impute_categorical(df, modes=None):
    df = df.copy()
    columns = df.select_dtypes(include=["object", "category"]).columns
    fitting = modes is None
    if fitting:
        modes = {}
        for col in columns:
            mode = df[col].mode()
            modes[col] = mode.iloc[0] if not mode.empty else "Unknown"

    filled = []
    for col in columns:
        n_nulls = int(df[col].isnull().sum())
        if n_nulls == 0:
            continue
        value = modes.get(col, "Unknown")
        df[col] = df[col].fillna(value)
        filled.append((col, n_nulls, str(value)))

    source = "Training Mode" if fitting else "Saved Training Mode"
    _print_fill_report("categorical", source, "Mode Used", filled, value_width=20)
    return df, modes


def _print_fill_report(kind, source, value_label, rows, value_width):
    if not rows:
        print(f"  No nulls found in {kind} columns.")
        return
    print(f"  {kind.capitalize()} Null Check — Filled with {source}\n")
    print(f"  {'Column':<30} {'Nulls':>6}  {value_label:>{value_width}}")
    print("  " + "-" * (30 + 1 + 6 + 2 + value_width))
    for col, count, value in rows:
        print(f"  {col:<30} {count:>6}  {value:>{value_width}}")
    print(f"\n  Total columns filled: {len(rows)}\n")


def split_features_target(df, target):
    """Numeric feature matrix (NaN → 0) and target series."""
    X = df.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(0)
    return X, df[target]


def save_artifacts(artifacts, save_dir):
    """Pickle every {name: object} pair to `<save_dir>/<name>.pkl`."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in artifacts.items():
        with open(save_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)
        print(f"✓  {name}")
    print(f"\nAll artefacts saved to ./{save_dir}/")
    print("Files:", sorted(p.name for p in save_dir.iterdir()))



# FEATURE ENGINEERING
_RAW_COUNT_COLUMNS = ["DemoCount", "PackageCount", "DeveloperCount",
                      "PublisherCount", "RequiredAge", "AchievementCount"]

_POPULARITY_COLUMNS = ["RecommendationCount", "Metacritic", "SteamSpyOwners",
                       "SteamSpyOwnersVariance", "SteamSpyPlayersEstimate",
                       "SteamSpyPlayersVariance", "AchievementHighlightedCount"]
_POPULARITY_SUMMARY_COLUMNS = ["RecommendationCount", "Metacritic",
                               "AchievementHighlightedCount", "relative_variation_owners",
                               "relative_variation_players", "Owners_to_Players_Ratio"]

# Interaction features that are log-transformed after creation.
_INTERACTION_LOG_FEATURES = ["Value_for_Money", "Marketing_Price_Impact", "Game_Momentum",
                             "Revenue_Proxy", "Quality_Visibility", "Price_Age_Penalty",
                             "Content_Engagement"]


def engineer_features(df, show_plots=True):
    """Run the full feature-engineering chain and return a new DataFrame.

    Steps (order matters — later steps use columns created by earlier ones):
      1. count / tier / pricing features
      2. text, language, release-date and hardware-requirement features
      3. popularity statistics (SteamSpy etc.)
      4. interaction features
    Finally the raw identifier / free-text columns are dropped.
    """
    df = df.copy()
    df = _add_count_and_pricing_features(df)
    df = _add_text_date_and_hardware_features(df)
    df = _add_popularity_features(df, show_plots)
    df = _add_interaction_features(df, show_plots)
    return df.drop(columns=cfg.COLUMNS_TO_DROP, errors="ignore")


def _has(df, *columns):
    return all(col in df.columns for col in columns)


#1. Counts, tiers and pricing 

def _count_tier(counts):
    """Bucket a count into 0 / 1 / 2+ (used for packages, developers, publishers)."""
    return np.select([counts == 0, counts == 1, counts >= 2], [0, 1, 2], default=1)


def _add_count_and_pricing_features(df):
    # Media assets / marketing
    df["Total_Media_Assets"] = np.log1p(df["ScreenshotCount"] + df["MovieCount"])
    df["ScreenshotCount_Log"] = np.log1p(df["ScreenshotCount"])
    df["MovieCount_Log"] = np.log1p(df["MovieCount"])
    df["Total_Media_Assets_log"] = np.log1p(df["Total_Media_Assets"])
    df["Marketing_Tier"] = np.select(
        [
            df["Total_Media_Assets"] == 0,
            (df["ScreenshotCount"] < 10) & (df["MovieCount"] <= 1),
            (df["ScreenshotCount"] <= 15) & (df["MovieCount"] <= 2),
        ],
        [0, 1, 2],
        default=3,
    )
    df["Is_Blockbuster"] = (df["MovieCount"] >= 3).astype(int)

    # Flags
    if "GenreIsNonGame" in df.columns:
        df["Is_NonGame_Flag"] = df["GenreIsNonGame"].astype(int)
    df["Zero_Owners_Flag"] = (df["SteamSpyOwners"] == 0).astype(int)
    df["Has_Demo"] = (df["DemoCount"] > 0).astype(int)

    # Count tiers
    df["Package_Tier"] = _count_tier(df["PackageCount"])
    df["PackageCount_Log"] = np.log1p(df["PackageCount"])
    df["Developer_Tier"] = _count_tier(df["DeveloperCount"])
    df["Publisher_Tier"] = _count_tier(df["PublisherCount"])
    df["Age_Tier"] = np.select(
        [
            df["RequiredAge"] == 0,
            (df["RequiredAge"] > 0) & (df["RequiredAge"] < 17),
            df["RequiredAge"] >= 17,
        ],
        [0, 1, 2],
        default=0,
    )
    df["Achievement_Tier"] = np.select(
        [
            df["AchievementCount"] == 0,
            (df["AchievementCount"] > 0) & (df["AchievementCount"] <= 50),
            (df["AchievementCount"] > 50) & (df["AchievementCount"] <= 150),
        ],
        [0, 1, 2],
        default=3,
    )
    df["AchievementCount_Log"] = np.log1p(df["AchievementCount"])

    # Pricing
    price = df["PriceFinal"]
    df["Price_Tier"] = np.select(
        [
            price == 0.0,
            (price > 0.0) & (price <= 5.0),
            (price > 5.0) & (price <= 15.0),
            (price > 15.0) & (price <= 40.0),
            price > 40.0,
        ],
        [0, 1, 2, 3, 4],
        default=0,
    )
    initial = df["PriceInitial"]
    discount = np.where(initial > 0, ((initial - price) / initial) * 100, 0.0)
    df["Discount_Percentage"] = np.round(discount, 2)
    df["Has_Discount"] = (df["Discount_Percentage"] > 0).astype(int)
    df["DLCCount_Log"] = np.log1p(df["DLCCount"])
    df["Has_DLC"] = (df["DLCCount"] > 0).astype(int)

    # Aggregates over related flag columns
    platform_cols = [c for c in ["PlatformWindows", "PlatformLinux", "PlatformMac"] if c in df.columns]
    if platform_cols:
        df["Platform_Reach"] = df[platform_cols].sum(axis=1)

    genre_cols = [c for c in df.columns if c.startswith("GenreIs")]
    if genre_cols:
        df["Genre_Diversity"] = df[genre_cols].sum(axis=1)

    engagement_cols = [c for c in ["CategoryMultiplayer", "CategoryCoop", "CategoryMMO"] if c in df.columns]
    if engagement_cols:
        df["Engagement_Score"] = df[engagement_cols].sum(axis=1)

    return df.drop(columns=_RAW_COUNT_COLUMNS)


# 2. Text, languages, release date, hardware requirements

def _add_text_date_and_hardware_features(df):
    df = _encode_binary_text_columns(df)
    df = _add_review_features(df)
    df = _add_text_length_features(df)
    df = _add_language_features(df)
    df = _cast_bool_to_int(df)
    df = _add_release_date_features(df)
    df = _encode_currency(df)
    df = _add_hardware_requirements(df)
    return df


def _is_non_empty(series):
    return series.fillna("").astype(str).str.strip().ne("").astype(int)


def _count_keywords(series, words):
    return series.apply(
        lambda text: sum(w in text.lower() for w in words) if isinstance(text, str) else 0
    )


def _encode_binary_text_columns(df):
    for col in cfg.BINARY_TEXT_COLUMNS:
        if col in df.columns:
            df[col] = _is_non_empty(df[col])
    return df


def _add_review_features(df):
    if "Reviews" not in df.columns:
        return df
    df["Review_Has_Content"] = _is_non_empty(df["Reviews"])
    df["Review_Positive_Signals"] = _count_keywords(df["Reviews"], cfg.POSITIVE_WORDS)
    df["Review_Negative_Signals"] = _count_keywords(df["Reviews"], cfg.NEGATIVE_WORDS)
    df["Review_Sentiment_Score"] = df["Review_Positive_Signals"] - df["Review_Negative_Signals"]
    return df.drop(columns=["Reviews"])


def _add_text_length_features(df):
    for col in cfg.TEXT_LENGTH_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.len()
    return df


def _add_language_features(df):
    if "SupportedLanguages" not in df.columns:
        return df
    languages = df["SupportedLanguages"]
    df["NumLanguages"] = languages.apply(
        lambda x: sum(1 for lang in cfg.COMMON_LANGUAGES if isinstance(x, str) and lang in x)
    )
    df["Has_Asian_Languages"] = languages.apply(
        lambda x: int(any(lang in str(x) for lang in cfg.ASIAN_LANGUAGES))
    )
    return df


def _cast_bool_to_int(df):
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
    return df


def _add_release_date_features(df):
    if "ReleaseDate" not in df.columns:
        return df
    dates = pd.to_datetime(df["ReleaseDate"], errors="coerce")
    dates = dates.fillna(dates.dropna().median())

    df["ReleaseDate_Year"] = dates.dt.year.astype("Int64")
    df["ReleaseDate_Month"] = dates.dt.month.astype("Int64")
    df["ReleaseDate_Day"] = dates.dt.day.astype("Int64")
    df["GameAge"] = datetime.now().year - df["ReleaseDate_Year"]
    df["Is_Holiday_Release"] = df["ReleaseDate_Month"].isin([11, 12]).astype(int)
    df["Release_Quarter"] = np.ceil(df["ReleaseDate_Month"] / 3).astype("Int64")
    return df


def _encode_currency(df):
    """1 if the price currency is USD (blank counts as USD), else 0."""
    if "PriceCurrency" in df.columns:
        currency = df["PriceCurrency"].astype(str).str.strip().replace("", "USD")
        df["PriceCurrency"] = (currency == "USD").astype(int)
    return df


def _to_base_unit(value, unit, small_unit):
    """Convert to GB / GHz: values given in the smaller unit (MB / MHz) are divided by 1000."""
    value = float(value)
    return value / 1000 if unit.lower() == small_unit else value


def _parse_requirements(text):
    """Extract RAM, storage, CPU speed and OpenGL version from a requirements string."""
    if not isinstance(text, str) or not text.strip():
        return dict.fromkeys(cfg.REQUIREMENT_FIELDS)

    ram = re.findall(r"(\d+)\s*(GB|mb)\s*(?:Memory|RAM)", text, re.IGNORECASE)
    storage = re.findall(r"(\d+)\s*GB\s*Hard\s*Drive", text, re.IGNORECASE)
    ghz = re.findall(r"(\d+\.?\d*)\s*(GHz|mhz)", text, re.IGNORECASE)
    opengl = re.findall(r"OpenGL\s*(\d+\.?\d*)", text, re.IGNORECASE)

    return {
        "RAM_GB": _to_base_unit(*ram[0], "mb") if ram else None,
        "Storage_GB": int(storage[0]) if storage else None,
        "CPU_GHz": _to_base_unit(*ghz[0], "mhz") if ghz else None,
        "OpenGL": float(opengl[0]) if opengl else None,
    }


def _add_hardware_requirements(df):
    for prefix, source in cfg.REQUIREMENT_SOURCES.items():
        if source in df.columns:
            parsed = df[source].apply(_parse_requirements).apply(pd.Series)
            parsed.columns = [f"{prefix}_{field}" for field in cfg.REQUIREMENT_FIELDS]
            df = pd.concat([df, parsed], axis=1)

    # Missing requirements are filled with the column minimum.
    for prefix in cfg.REQUIREMENT_SOURCES:
        for field in cfg.REQUIREMENT_FIELDS:
            col = f"{prefix}_{field}"
            if col in df.columns and df[col].notna().any():
                df[col] = df[col].fillna(df[col].min())
    return df


# 3. Popularity statistics

def _log_ratio(numerator, denominator, epsilon=0.0):
    """log1p(numerator / denominator), or 0 where the denominator is not positive."""
    return np.where(denominator > 0, np.log1p(numerator / (denominator + epsilon)), 0.0)


def _add_popularity_features(df, show_plots):
    present = [c for c in _POPULARITY_COLUMNS if c in df.columns]
    if show_plots and present:
        plot_distributions(df, present)

    # Ratios are computed on raw values, before the log transform below.
    if _has(df, "SteamSpyOwnersVariance", "SteamSpyOwners"):
        df["relative_variation_owners"] = _log_ratio(df["SteamSpyOwnersVariance"], df["SteamSpyOwners"])
    if _has(df, "SteamSpyPlayersVariance", "SteamSpyPlayersEstimate"):
        df["relative_variation_players"] = _log_ratio(df["SteamSpyPlayersVariance"], df["SteamSpyPlayersEstimate"])
    if _has(df, "SteamSpyOwners", "SteamSpyPlayersEstimate"):
        df["Owners_to_Players_Ratio"] = _log_ratio(
            df["SteamSpyPlayersEstimate"], df["SteamSpyOwners"], epsilon=1e-9
        )

    for col in present:
        if col != "AchievementHighlightedCount":
            df[col] = np.log1p(df[col])

    if "AchievementHighlightedCount" in df.columns:
        highlighted = df["AchievementHighlightedCount"]
        df["AchievementHighlightedCount"] = np.select(
            [highlighted == 0, highlighted == 10], [0, 2], default=1
        )

    summary = [c for c in _POPULARITY_SUMMARY_COLUMNS if c in df.columns]
    if show_plots and len(summary) > 1:
        corr_heatmap(df, summary)
    return df


# 4. Interaction features 

def _add_interaction_features(df, show_plots):
    print_header("Adding Interaction Features...")

    if _has(df, "Total_Media_Assets", "PriceFinal"):
        df["Value_for_Money"] = df["Total_Media_Assets"] / (df["PriceFinal"] + 1)
    if _has(df, "Marketing_Tier", "PriceFinal"):
        df["Marketing_Price_Impact"] = df["Marketing_Tier"] * df["PriceFinal"]
    if _has(df, "AchievementCount_Log", "GameAge"):
        df["Game_Momentum"] = df["AchievementCount_Log"] / (df["GameAge"] + 1)
    if _has(df, "SteamSpyOwners", "PriceFinal"):
        df["Revenue_Proxy"] = df["SteamSpyOwners"] * np.log1p(df["PriceFinal"])
    if _has(df, "Total_Media_Assets", "Metacritic"):
        df["Quality_Visibility"] = df["Total_Media_Assets"] * df["Metacritic"]
    if _has(df, "PriceFinal", "GameAge"):
        df["Price_Age_Penalty"] = df["PriceFinal"] / (df["GameAge"] + 1)
    if _has(df, "DLCCount_Log", "Engagement_Score"):
        df["Content_Engagement"] = df["DLCCount_Log"] * (df["Engagement_Score"] + 1)

    if _has(df, "PriceFinal", "PriceInitial"):
        df["price"] = df["PriceFinal"] + df["PriceInitial"]
        df["IsFree"] = (df["price"] == 0).astype(int)
        df["FreeVerAvail"] = (df["price"] == 0).astype(int)
        df["price"] = np.log1p(df["price"])

    for col in _INTERACTION_LOG_FEATURES:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    heatmap_cols = [c for c in ["RecommendationCount", *_INTERACTION_LOG_FEATURES] if c in df.columns]
    if show_plots and len(heatmap_cols) > 1:
        corr_heatmap(df, heatmap_cols)
    return df
