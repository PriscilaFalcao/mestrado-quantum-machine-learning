from __future__ import annotations

import time

from data.preprocessing import prepare_data
from evaluation.regression_metrics import mae, rmse
from experiments.common import read_yaml, save_json
from experiments.model_factory import build_model


def main() -> None:
    dataset_cfg = read_yaml("configs/dataset.yaml")
    models_cfg = read_yaml("configs/models.yaml")["models"]

    prepared = prepare_data(dataset_cfg["dataset"], random_seed=int(dataset_cfg["dataset"]["random_seed"]))
    splits = prepared.splits

    results = {}
    for model_name in ["mlp", "vqc"]:
        model = build_model(model_name, models_cfg, seed=int(dataset_cfg["dataset"]["random_seed"]))

        t0 = time.perf_counter()
        model.fit(splits.X_train, splits.y_train)
        train_seconds = time.perf_counter() - t0

        pred_test = model.predict(splits.X_test)

        model_result = {
            "mae": mae(splits.y_test, pred_test),
            "rmse": rmse(splits.y_test, pred_test),
            "train_seconds": train_seconds,
        }

        if model_name == "vqc":
            model_result["circuit_evaluations"] = int(getattr(model, "circuit_evaluations", 0))

        results[model_name] = model_result

    save_json("results/metrics/clean_baseline.json", results)
    print("Saved: results/metrics/clean_baseline.json")


if __name__ == "__main__":
    main()
