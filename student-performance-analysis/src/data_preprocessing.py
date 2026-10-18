"""Data loading, cleaning checks, splitting and the leakage-safe preprocessing pipeline."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from config import (DATA_PATH, TARGET, SEED, BINARY_COLS, NOMINAL_COLS, CLASS_NAMES)


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the UCI 'Predict Students' Dropout and Academic Success' CSV (';' separated)."""
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()          # header contains stray whitespace/tab
    return df


def numeric_cols(df: pd.DataFrame):
    """Numeric features = all features that are neither binary flags nor nominal codes."""
    return [c for c in df.columns if c not in BINARY_COLS + NOMINAL_COLS + [TARGET]]


def clean_data(df: pd.DataFrame):
    """Remove duplicates; report missing values. Returns (clean_df, report dict)."""
    report = {"rows_before": len(df),
              "missing_cells": int(df.isna().sum().sum()),
              "duplicate_rows": int(df.duplicated().sum())}
    df = df.drop_duplicates().reset_index(drop=True)
    report["rows_after"] = len(df)
    return df, report


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to its [lower, upper] percentile, learned on TRAINING data only."""
    def __init__(self, lower=1, upper=99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.percentile(X, self.lower, axis=0)
        self.hi_ = np.percentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, names=None):
        return np.asarray(names)


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """numeric -> winsorise -> standardise | nominal -> one-hot (rare levels grouped) | binary -> as is."""
    num = numeric_cols(df)
    return ColumnTransformer([
        ("num", Pipeline([("clip", Winsorizer(1, 99)), ("scale", StandardScaler())]), num),
        ("cat", OneHotEncoder(min_frequency=0.02, handle_unknown="infrequent_if_exist",
                              sparse_output=False), NOMINAL_COLS),
        ("bin", "passthrough", BINARY_COLS),
    ])


def split_data(df: pd.DataFrame, val_size=0.15, test_size=0.15):
    """Stratified 70/15/15 split of raw features; target encoded with a fixed class order."""
    le = LabelEncoder().fit(CLASS_NAMES)
    X, y = df.drop(columns=[TARGET]), pd.Series(le.transform(df[TARGET]), index=df.index)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=val_size + test_size, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=test_size / (val_size + test_size), stratify=y_tmp, random_state=SEED)
    return X_tr, X_val, X_te, y_tr, y_val, y_te
