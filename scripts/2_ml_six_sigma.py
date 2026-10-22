import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "tms_shipments.csv"
OUTPUT_PATH = ROOT / "outputs" / "rca_metrics.json"


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_model(df: pd.DataFrame):
    feature_cols = [
        "carrier",
        "region",
        "mode",
        "shipment_type",
        "service_level",
        "priority",
        "distance_km",
        "weight_tons",
        "weather_delay",
        "customs_delay",
        "dock_congestion",
        "driver_shortage",
        "late_pickup",
        "handling_error",
    ]
    target_col = "on_time"

    X = df[feature_cols]
    y = df[target_col]

    categorical_features = [col for col in feature_cols if not is_numeric_dtype(X[col])]
    numeric_features = [col for col in feature_cols if is_numeric_dtype(X[col])]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, objective="binary:logistic", eval_metric="auc", random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    return model, report, auc


def six_sigma_pareto(df: pd.DataFrame) -> dict:
    late_shipments = df[df["on_time"] == 0]
    causes = [
        "weather_delay",
        "customs_delay",
        "dock_congestion",
        "driver_shortage",
        "late_pickup",
        "handling_error",
    ]
    cause_scores = {}
    for cause in causes:
        score = late_shipments[cause].mean()
        cause_scores[cause] = round(float(score), 4)

    total = sum(cause_scores.values())
    pareto = []
    cumulative = 0.0
    for name, value in sorted(cause_scores.items(), key=lambda item: item[1], reverse=True):
        cumulative += value / total if total else 0.0
        pareto.append({"cause": name, "score": round(value, 4), "cumulative_share": round(cumulative, 4)})

    top_causes = [entry["cause"] for entry in pareto if entry["cumulative_share"] <= 0.8]
    return {"cause_scores": cause_scores, "pareto": pareto, "top_causes": top_causes}


def main() -> None:
    df = load_data()
    model, report, auc = build_model(df)
    pareto = six_sigma_pareto(df)

    metrics = {
        "sample_size": int(len(df)),
        "on_time_rate": round(float((df["on_time"] == 1).mean()), 4),
        "late_rate": round(float((df["on_time"] == 0).mean()), 4),
        "average_delay_hours": round(float(df["delay_hours"].mean()), 2),
        "model_auc": round(float(auc), 4),
        "classification_report": report,
        "six_sigma_pareto": pareto,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"Saved RCA metrics to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
