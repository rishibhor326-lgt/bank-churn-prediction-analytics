"""Evaluate reproducible churn baselines without touching the deployed model."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "Num Of Products", "Has Credit Card", "Is Active Member", "Estimated Salary",
]


def make_pipeline(classifier):
    return Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", StandardScaler(), ["CreditScore", "Age", "Tenure", "Balance", "Num Of Products", "Estimated Salary"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Geography", "Gender"]),
            ("binary", "passthrough", ["Has Credit Card", "Is Active Member"]),
        ])),
        ("classifier", classifier),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Churn_Modelling.csv"))
    parser.add_argument("--output", type=Path, default=Path("churn_model_candidate.pkl"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    target = "Churn" if "Churn" in df else "Exited"
    if target not in df or df[target].isna().any() or not df[target].isin([0, 1]).all():
        raise ValueError("Expected a complete binary Churn or Exited target")
    if df[FEATURES].isna().any().any():
        raise ValueError("Model inputs contain missing values; inspect the data before training")
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[target], test_size=0.2, random_state=42, stratify=df[target]
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    print(f"Rows: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
    results = {}
    for name, classifier in models.items():
        model = make_pipeline(classifier)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="binary", zero_division=0
        )
        results[name] = {
            "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y_test, predictions, labels=[0, 1]).ravel().tolist(),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        }
        if name == "Random Forest":
            joblib.dump(model, args.output)
    print(json.dumps(results, indent=2))
    print(f"Candidate saved to {args.output}; deployed churn_model.pkl was not changed")


if __name__ == "__main__":
    main()
