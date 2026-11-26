from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from data.splits import DatasetSplits, create_regression_splits


@dataclass
class PreparedData:
    splits: DatasetSplits
    feature_scaler: StandardScaler | None
    target_scaler: StandardScaler | None
    feature_names: list[str]



def _load_csv_dataset(
    csv_path: str, target_column: str, drop_columns: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    if drop_columns:
        # Names in the CSV may carry stray spaces; match on the stripped name.
        wanted = {c.strip() for c in drop_columns}
        df = df.drop(columns=[c for c in df.columns if c.strip() in wanted and c != target_column])
DEFAULT_DATE_PARTS = ["year", "month", "day", "dayofweek", "dayofyear"]


def _expand_date_columns(
    df: pd.DataFrame,
    date_columns: list[str],
    date_format: str,
    dayfirst: bool,
    date_parts: list[str],
) -> pd.DataFrame:
    for col in date_columns:
        if col not in df.columns:
            raise ValueError(f"Date column '{col}' not found in CSV.")
        parsed = pd.to_datetime(df[col], format=date_format, dayfirst=dayfirst, errors="coerce")
        if parsed.isna().all():
            raise ValueError(f"Could not parse any value of date column '{col}'.")
        for part in date_parts:
            values = getattr(parsed.dt, part)
            df[f"{col}_{part}"] = values.astype(float)
        df = df.drop(columns=[col])
    return df


def _load_csv_dataset(dataset_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    csv_path = dataset_cfg["csv_path"]
    target_column = dataset_cfg["target_column"]

    path = Path(csv_path)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).resolve().parents[1] / path

    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV.")

    df = df.dropna(subset=[target_column])
    y = df[target_column].to_numpy(dtype=float)
    features = df.drop(columns=[target_column])

    drop_columns = list(dataset_cfg.get("drop_columns") or [])
    missing = [c for c in drop_columns if c not in features.columns]
    if missing:
        raise ValueError(f"drop_columns not found in CSV: {missing}")
    features = features.drop(columns=drop_columns)

    date_columns = list(dataset_cfg.get("date_columns") or [])
    if date_columns:
        features = _expand_date_columns(
            features,
            date_columns=date_columns,
            date_format=str(dataset_cfg.get("date_format", "mixed")),
            dayfirst=bool(dataset_cfg.get("date_dayfirst", True)),
            date_parts=list(dataset_cfg.get("date_parts") or DEFAULT_DATE_PARTS),
        )

    categorical_columns = list(dataset_cfg.get("categorical_columns") or [])
    if categorical_columns:
        missing = [c for c in categorical_columns if c not in features.columns]
        if missing:
            raise ValueError(f"categorical_columns not found in CSV: {missing}")
        # Category vocabulary is taken from the full CSV so feature names/indices are stable across seeds.
        features = pd.get_dummies(
            features,
            columns=categorical_columns,
            drop_first=bool(dataset_cfg.get("one_hot_drop_first", False)),
            dtype=float,
        )

    features = features.select_dtypes(include=[np.number])
    X = features.to_numpy(dtype=float)
    feature_names = list(features.columns)

    if X.size == 0:
        raise ValueError("No numeric features available after preprocessing CSV.")

    max_samples = dataset_cfg.get("max_samples")
    if max_samples is not None and len(X) > int(max_samples):
        rng = np.random.default_rng(int(dataset_cfg.get("random_seed", 42)))
        idx = np.sort(rng.choice(len(X), size=int(max_samples), replace=False))
        X, y = X[idx], y[idx]

    return X, y, feature_names


def load_dataset_from_config(dataset_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    name = dataset_cfg.get("name", "diabetes")
    csv_path = dataset_cfg.get("csv_path")
    target_column = dataset_cfg.get("target_column")

    if csv_path:
        if not target_column:
            raise ValueError("target_column is required when csv_path is provided.")
        return _load_csv_dataset(dataset_cfg)

    if name == "diabetes":
        ds = load_diabetes()
        return ds.data.astype(float), ds.target.astype(float), list(ds.feature_names)

    raise ValueError(f"Unsupported built-in dataset: {name}")


def prepare_data(dataset_cfg: dict[str, Any], random_seed: int | None = None) -> PreparedData:
    X, y, feature_names = load_dataset_from_config(dataset_cfg)

    if dataset_cfg.get("log_target", False):
        # Heavy-tailed targets (e.g. GDP) are modelled on the log scale; log1p keeps zeros valid.
        if np.any(y < 0):
            raise ValueError("log_target requires a non-negative target.")
        y = np.log1p(y)

    test_size = float(dataset_cfg.get("test_size", 0.2))
    val_size = float(dataset_cfg.get("val_size", 0.2))
    seed = int(dataset_cfg.get("random_seed", 42) if random_seed is None else random_seed)

    splits = create_regression_splits(
        X=X,
        y=y,
        test_size=test_size,
        val_size=val_size,
        random_seed=seed,
    )

    if dataset_cfg.get("log_features", False):
        # Signed log1p on the heavy-tailed columns. They are chosen on the training split only
        # (|skewness| above `log_skew_threshold`), then the same columns are transformed everywhere.
        threshold = float(dataset_cfg.get("log_skew_threshold", 1.0))
        train = splits.X_train
        skews = np.array([skew(train[:, j], nan_policy="omit") for j in range(train.shape[1])])
        log_cols = np.where(np.abs(skews) > threshold)[0]
        for part in ("X_train", "X_val", "X_test"):
            arr = getattr(splits, part).copy()
            arr[:, log_cols] = np.sign(arr[:, log_cols]) * np.log1p(np.abs(arr[:, log_cols]))
            setattr(splits, part, arr)

    # Fit only on train to avoid leakage and apply consistently across splits.
    imputer = SimpleImputer(strategy="median")
    splits.X_train = imputer.fit_transform(splits.X_train)
    splits.X_val = imputer.transform(splits.X_val)
    splits.X_test = imputer.transform(splits.X_test)

    standardize_features = bool(dataset_cfg.get("standardize_features", True))
    standardize_target = bool(dataset_cfg.get("standardize_target", True))

    feature_scaler: StandardScaler | None = None
    target_scaler: StandardScaler | None = None

    if standardize_features:
        feature_scaler = StandardScaler()
        splits.X_train = feature_scaler.fit_transform(splits.X_train)
        splits.X_val = feature_scaler.transform(splits.X_val)
        splits.X_test = feature_scaler.transform(splits.X_test)

    if standardize_target:
        target_scaler = StandardScaler()
        splits.y_train = target_scaler.fit_transform(splits.y_train.reshape(-1, 1)).ravel()
        splits.y_val = target_scaler.transform(splits.y_val.reshape(-1, 1)).ravel()
        splits.y_test = target_scaler.transform(splits.y_test.reshape(-1, 1)).ravel()

    return PreparedData(
        splits=splits,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_names=feature_names,
    )
