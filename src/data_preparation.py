"""Data loading, cleaning, encoding, feature engineering, splitting, and scaling."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from src.utils import RANDOM_STATE


def load_data(filepath="data/bank-additional-full.csv"):
    """Load the Bank Marketing dataset."""
    df = pd.read_csv(filepath, sep=";")
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def inspect_data(df):
    """Print dataset overview and missing value summary."""
    print("\n--- Data Types ---")
    print(df.dtypes)
    print("\n--- Descriptive Statistics ---")
    print(df.describe(include="all"))
    print("\n--- 'unknown' counts per column ---")
    unknown_counts = {}
    for col in df.select_dtypes(include="object").columns:
        count = (df[col] == "unknown").sum()
        if count > 0:
            unknown_counts[col] = (count, f"{count / len(df) * 100:.2f}%")
    for col, (count, pct) in sorted(unknown_counts.items(), key=lambda x: -x[1][0]):
        print(f"  {col}: {count} ({pct})")
    return unknown_counts


def clean_data(df):
    """Handle missing values and special encodings.

    - Drop 'default' column (20.87% unknown, only 3 'yes' values — near-zero variance).
    - Replace 'unknown' with NaN for later imputation.
    - Transform pdays: create 'was_previously_contacted' binary feature,
      set pdays=0 for non-contacted clients.
    """
    df = df.copy()
    df = df.drop(columns=["default"])
    df = df.replace("unknown", np.nan)
    df["was_previously_contacted"] = (df["pdays"] != 999).astype(int)
    df.loc[df["pdays"] == 999, "pdays"] = 0
    df["y"] = (df["y"] == "yes").astype(int)
    return df


def impute_missing(df_train, df_val, df_test):
    """Impute categorical NaNs with mode from training set only (no data leakage).

    Features imputed: job, marital, education, housing, loan.
    """
    cat_cols_to_impute = ["job", "marital", "education", "housing", "loan"]
    modes = {}
    for col in cat_cols_to_impute:
        if col in df_train.columns:
            modes[col] = df_train[col].mode()[0]

    for df in [df_train, df_val, df_test]:
        for col, mode_val in modes.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode_val)

    print(f"Imputed columns with training modes: {modes}")
    return df_train, df_val, df_test


def encode_features(df_train, df_val, df_test):
    """Encode categorical features.

    - education: ordinal (natural ordering exists)
    - contact: label encode (binary)
    - job, marital, month, day_of_week, poutcome: one-hot
    """
    edu_order = [
        "illiterate", "basic.4y", "basic.6y", "basic.9y",
        "high.school", "professional.course", "university.degree"
    ]
    for df in [df_train, df_val, df_test]:
        df["education"] = df["education"].map(
            {level: i for i, level in enumerate(edu_order)}
        )

    for df in [df_train, df_val, df_test]:
        df["contact"] = (df["contact"] == "cellular").astype(int)

    for df in [df_train, df_val, df_test]:
        df["housing"] = (df["housing"] == "yes").astype(int)
        df["loan"] = (df["loan"] == "yes").astype(int)

    ohe_cols = ["job", "marital", "month", "day_of_week", "poutcome"]
    df_train = pd.get_dummies(df_train, columns=ohe_cols, drop_first=True, dtype=int)
    df_val = pd.get_dummies(df_val, columns=ohe_cols, drop_first=True, dtype=int)
    df_test = pd.get_dummies(df_test, columns=ohe_cols, drop_first=True, dtype=int)

    # Align columns across splits (handle unseen categories)
    for col in df_train.columns:
        if col not in df_val.columns:
            df_val[col] = 0
        if col not in df_test.columns:
            df_test[col] = 0
    df_val = df_val[df_train.columns]
    df_test = df_test[df_train.columns]

    print(f"Encoded features. Final shape: {df_train.shape[1]} columns")
    return df_train, df_val, df_test


def engineer_features(df):
    """Create new features before splitting.

    - contact_rate: campaign / (previous + 1)
    """
    df = df.copy()
    df["contact_rate"] = df["campaign"] / (df["previous"] + 1)
    return df


def split_data(df, target_col="y", train_size=0.70, val_size=0.15, test_size=0.15):
    """Split into train/val/test with stratification.

    70/15/15 split. Stratified to preserve class ratios.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    val_relative = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=RANDOM_STATE
    )

    print(f"Split sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Positive rate — Train: {y_train.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """Standardize features for MLP. Fit on train only (no data leakage).

    Returns scaled copies and the fitted scaler.
    """
    scaler = StandardScaler()
    feature_cols = X_train.columns

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=feature_cols, index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )

    print("Features standardized (fit on train only)")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def prepare_data(filepath="data/bank-additional-full.csv"):
    """Full data preparation pipeline. Returns all splits for both GBDT and MLP.

    Returns:
        dict with keys:
            'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test' — for GBDT
            'X_train_scaled', 'X_val_scaled', 'X_test_scaled' — for MLP
            'scaler' — fitted StandardScaler
            'feature_names' — list of feature column names
    """
    df = load_data(filepath)
    inspect_data(df)
    df = clean_data(df)
    df = engineer_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train, X_val, X_test = impute_missing(X_train, X_val, X_test)
    X_train, X_val, X_test = encode_features(X_train, X_val, X_test)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test
    )

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "feature_names": list(X_train.columns),
    }
