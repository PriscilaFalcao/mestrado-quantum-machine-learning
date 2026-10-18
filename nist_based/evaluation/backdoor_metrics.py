from __future__ import annotations

import numpy as np

from evaluation.regression_metrics import mae, rmse


def mean_prediction_shift(y_clean_pred: np.ndarray, y_trigger_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_trigger_pred - y_clean_pred)))


def backdoor_success_rate(
    y_trigger_pred: np.ndarray,
    target_value: float,
    tolerance: float,
) -> float:
    success = np.abs(y_trigger_pred - target_value) <= tolerance
    return float(np.mean(success))


def evaluate_backdoor(
    y_true_clean: np.ndarray,
    y_clean_pred: np.ndarray,
    y_trigger_pred: np.ndarray,
    target_value: float,
    tolerance: float,
) -> dict[str, float]:
    return {
        "clean_mae": mae(y_true_clean, y_clean_pred),
        "clean_rmse": rmse(y_true_clean, y_clean_pred),
        "trigger_mae": mae(y_true_clean, y_trigger_pred),
        "trigger_rmse": rmse(y_true_clean, y_trigger_pred),
        "prediction_shift": mean_prediction_shift(y_clean_pred, y_trigger_pred),
        "backdoor_success_rate": backdoor_success_rate(
            y_trigger_pred=y_trigger_pred,
            target_value=target_value,
            tolerance=tolerance,
        ),
    }
