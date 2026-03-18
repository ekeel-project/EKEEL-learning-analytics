#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WP4 Learning Analytics — Mode 4 Outcome Prediction (Random Forest).

This script mirrors the predictive analysis described in D4.2 for Mode 4 (IKCM):
- loads the feature table produced by the Mode 4 pipeline (Mode4Features.csv)
- trains a RandomForestRegressor to predict score_percent
- exports metrics and feature importances

Usage
-----
python mode4_predict_rf.py --features_csv <Report/Mode4Features.csv> --out_dir <Report/Predictive>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


def run(features_csv: Path, out_dir: Path, target: str = "score_percent", random_state: int = 42) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {features_csv}. Available columns: {list(df.columns)[:20]}...")

    # Keep only numeric predictors
    X = df.drop(columns=[target]).select_dtypes(include=["number"]).copy()
    y = df[target].astype(float).copy()

    # Basic missing handling
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))

    metrics = pd.DataFrame([
        {"target": target, "n": len(df), "n_features": X.shape[1], "r2": r2, "mse": mse, "rmse": rmse}
    ])
    metrics.to_csv(out_dir / "rf_metrics.csv", index=False)

    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importances.to_csv(out_dir / "rf_feature_importance.csv", index=False)

    # Plot top-20 importances
    topk = importances.head(20).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(topk["feature"], topk["importance"])
    plt.xlabel("Importance")
    plt.title("Random Forest feature importance (top-20)")
    plt.tight_layout()
    plt.savefig(out_dir / "rf_feature_importance_top20.png", dpi=200)
    plt.close()

    # Minimal HTML report
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Mode 4 — Outcome Prediction (RF)</title></head>
<body>
<h1>Mode 4 — Outcome Prediction (Random Forest)</h1>
<p><b>Input</b>: {features_csv}</p>
<h2>Metrics</h2>
<pre>{metrics.to_string(index=False)}</pre>
<h2>Top feature importances</h2>
<p><img src='rf_feature_importance_top20.png' style='max-width: 100%; height: auto;'></p>
<p>Full importances: <code>rf_feature_importance.csv</code></p>
</body></html>"""

    (out_dir / "rf_report.html").write_text(html, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="WP4 — Mode 4 outcome prediction (Random Forest)")
    p.add_argument("--features_csv", type=str, required=True, help="Path to Mode4Features.csv produced by the Mode 4 pipeline")
    p.add_argument("--out_dir", type=str, required=True, help="Output folder")
    p.add_argument("--target", type=str, default="score_percent", help="Target column (default: score_percent)")
    args = p.parse_args()

    run(Path(args.features_csv), Path(args.out_dir), target=args.target)


if __name__ == "__main__":
    main()
