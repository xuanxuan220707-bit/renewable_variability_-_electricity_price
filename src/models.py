"""Chronological classification, ablation, uncertainty, and alarm trade-offs."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import chronological_masks

SEED = 42


def classifier(kind: str):
    if kind == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=1500, random_state=SEED))
    if kind == "forest":
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=20, max_features=.8,
            n_jobs=2, random_state=SEED))
    return make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(
        max_iter=160, max_leaf_nodes=15, learning_rate=.06, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=False, random_state=SEED))


def choose_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    grid = np.linspace(.01, .99, 99)
    values = [f1_score(y, scores >= threshold, zero_division=0) for threshold in grid]
    return float(grid[int(np.argmax(values))])


def metric_row(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    alarm = scores >= threshold
    return {"AP": average_precision_score(y, scores), "ROC_AUC": roc_auc_score(y, scores),
            "Brier": brier_score_loss(y, scores), "Precision": precision_score(y, alarm, zero_division=0),
            "Recall": recall_score(y, alarm, zero_division=0), "F1": f1_score(y, alarm, zero_division=0),
            "Alarm_rate": float(alarm.mean())}


def train_and_evaluate(features: pd.DataFrame, groups: dict, root: Path) -> dict:
    masks = chronological_masks(features)
    all_columns = sum(groups.values(), [])
    configurations = {
        "Calendar logistic": ("logistic", groups["Calendar"]),
        "Price + calendar": ("logistic", groups["Price history"] + groups["Calendar"]),
        "Full logistic": ("logistic", all_columns),
        "Full forest": ("forest", all_columns),
        "Full boosting": ("boosting", all_columns),
        "Boosting without weather": ("boosting", [c for c in all_columns if c not in groups["Weather"]]),
        "Boosting without renewables": ("boosting", [c for c in all_columns if c not in groups["Renewables and residual load"]]),
    }
    models, probabilities, thresholds, rows = {}, {}, {}, []
    split_rows = []
    for split, mask in masks.items():
        part = features.loc[mask]
        split_rows.append({"Split": split, "Origins": len(part), "Events": int(part.target.sum()),
                           "Event_rate": float(part.target.mean()), "First_origin": str(part.index.min()),
                           "Last_origin": str(part.index.max()), "Last_target": str(part.target_time.max())})
    split_table = pd.DataFrame(split_rows)
    split_table.to_csv(root / "outputs" / "split_summary.csv", index=False)
    with threadpool_limits(limits=2):
        for name, (kind, columns) in configurations.items():
            model = classifier(kind)
            model.fit(features.loc[masks["train"], columns], features.loc[masks["train"], "target"])
            models[name] = (model, columns)
            probabilities[name] = {split: model.predict_proba(features.loc[masks[split], columns])[:, 1]
                                   for split in ["validation", "test"]}
            threshold = choose_threshold(features.loc[masks["validation"], "target"].to_numpy(), probabilities[name]["validation"])
            thresholds[name] = threshold
            for split in ["validation", "test"]:
                values = metric_row(features.loc[masks[split], "target"].to_numpy(), probabilities[name][split], threshold)
                rows.append({"Model": name, "Split": split, "Threshold": threshold, **values})
            print(f"Fitted {name}", flush=True)
    metrics = pd.DataFrame(rows)
    # Selection uses validation data only; ablations diagnose the fixed boosting architecture.
    candidates = ["Price + calendar", "Full logistic", "Full forest", "Full boosting"]
    validation = metrics[(metrics.Split == "validation") & metrics.Model.isin(candidates)]
    selected = str(validation.loc[validation.AP.idxmax(), "Model"])
    metrics.to_csv(root / "outputs" / "model_metrics.csv", index=False)
    test = features.loc[masks["test"]]
    predictions = pd.DataFrame({"origin_time": test.index, "target_time": test.target_time.to_numpy(),
                                "target_price": test.target_price.to_numpy(), "high_price": test.target.to_numpy()})
    for name in configurations:
        predictions[name] = probabilities[name]["test"]
    predictions["alarm"] = probabilities[selected]["test"] >= thresholds[selected]
    predictions.to_csv(root / "outputs" / "test_predictions.csv", index=False)
    model, columns = models[selected]
    joblib.dump({"model": model, "columns": columns, "threshold": thresholds[selected],
                 "horizon_hours": 6, "selected_model": selected,
                 "price_event_threshold": json.loads((root / "outputs" / "risk_thresholds.json").read_text())["price_q90"]},
                root / "outputs" / "warning_model.joblib")
    (root / "outputs" / "model_selection.json").write_text(json.dumps({"selected_model": selected,
        "criterion": "highest validation average precision among four forecasting candidates",
        "threshold_selection": "maximum validation F1 on 0.01--0.99 grid",
        "seed": SEED, "thresholds": thresholds,
        "settings": {"logistic_C": 1.0, "boosting_iterations": 160, "boosting_learning_rate": .06,
                     "boosting_leaves": 15, "forest_trees": 150, "training_refit_after_validation": False}}, indent=2)+"\n")
    return {"features": features, "groups": groups, "masks": masks, "models": models,
            "probabilities": probabilities, "thresholds": thresholds, "metrics": metrics,
            "selected": selected, "splits": split_table, "predictions": predictions}


def bootstrap_comparison(result: dict, root: Path, repetitions: int = 250) -> pd.DataFrame:
    test = result["features"].loc[result["masks"]["test"]]
    y = test.target.to_numpy()
    selected = result["selected"]
    p = result["probabilities"][selected]["test"]
    baseline = result["probabilities"]["Price + calendar"]["test"]
    # Seven-day blocks retain short-range serial dependence and pair model predictions.
    labels = ((test.index - test.index.min()) // pd.Timedelta(days=7)).to_numpy()
    blocks = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(repetitions):
        ids = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        if len(np.unique(y[ids])) < 2:
            continue
        model_ap = average_precision_score(y[ids], p[ids])
        baseline_ap = average_precision_score(y[ids], baseline[ids])
        samples.append([model_ap, baseline_ap, model_ap - baseline_ap])
    point = [average_precision_score(y, p), average_precision_score(y, baseline),
             average_precision_score(y, p) - average_precision_score(y, baseline)]
    samples = np.asarray(samples)
    table = pd.DataFrame({"Quantity": ["Selected model AP", "Price + calendar AP", "Paired AP difference"],
                          "Estimate": point, "Lower_95": np.quantile(samples, .025, axis=0),
                          "Upper_95": np.quantile(samples, .975, axis=0)})
    table.to_csv(root / "outputs" / "bootstrap_comparison.csv", index=False)
    return table


def grouped_importance(result: dict, root: Path, repetitions: int = 10) -> pd.DataFrame:
    test = result["features"].loc[result["masks"]["test"]]
    # Inspect the full boosting model even if another candidate wins validation selection.
    name = "Full boosting"
    model, columns = result["models"][name]
    x = test[columns]
    y = test.target.to_numpy()
    reference = average_precision_score(y, result["probabilities"][name]["test"])
    rng = np.random.default_rng(SEED)
    rows = []
    with threadpool_limits(limits=2):
        for group, group_columns in result["groups"].items():
            differences = []
            for _ in range(repetitions):
                shifted = x.copy()
                offset = int(rng.integers(7, max(8, len(x)//24 - 7))) * 24
                shifted[group_columns] = np.roll(x[group_columns].to_numpy(), offset, axis=0)
                score = average_precision_score(y, model.predict_proba(shifted)[:, 1])
                differences.append(reference - score)
            rows.append({"Group": group, "AP_drop": np.mean(differences), "SD": np.std(differences, ddof=1)})
    table = pd.DataFrame(rows).sort_values("AP_drop", ascending=False)
    table.to_csv(root / "outputs" / "group_importance.csv", index=False)
    return table


def alarm_tradeoffs(result: dict, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = result["selected"]
    features = result["features"]
    yv = features.loc[result["masks"]["validation"], "target"].to_numpy()
    yt = features.loc[result["masks"]["test"], "target"].to_numpy()
    pv = result["probabilities"][selected]["validation"]
    pt = result["probabilities"][selected]["test"]
    grid = np.linspace(.01, .99, 99)
    curve = []
    for threshold in grid:
        values = metric_row(yt, pt, float(threshold))
        tn, fp, fn, tp = confusion_matrix(yt, pt >= threshold, labels=[0, 1]).ravel()
        curve.append({"Threshold": threshold, "False_alarms_per_1000": fp/len(yt)*1000,
                      "Misses_per_1000": fn/len(yt)*1000, **values})
    decisions = []
    for ratio in [1, 3, 5, 10, 20]:
        losses = []
        for threshold in grid:
            tn, fp, fn, tp = confusion_matrix(yv, pv >= threshold, labels=[0,1]).ravel()
            losses.append((fp + ratio*fn) / len(yv))
        threshold = float(grid[int(np.argmin(losses))])
        tn, fp, fn, tp = confusion_matrix(yt, pt >= threshold, labels=[0,1]).ravel()
        decisions.append({"Miss_cost_ratio": ratio, "Threshold": threshold,
                          "Test_relative_loss": (fp+ratio*fn)/len(yt),
                          "False_alarms_per_1000": fp/len(yt)*1000, "Misses_per_1000": fn/len(yt)*1000,
                          "Alarm_rate": float(np.mean(pt >= threshold)), "Recall": tp/(tp+fn)})
    curve = pd.DataFrame(curve)
    decisions = pd.DataFrame(decisions)
    curve.to_csv(root / "outputs" / "alarm_threshold_curve.csv", index=False)
    decisions.to_csv(root / "outputs" / "decision_scenarios.csv", index=False)
    return curve, decisions
