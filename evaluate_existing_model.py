"""Read-only check of the saved app model on a reconstructed 20% split.

The training split used for the saved artifact is not recorded. This script
recreates a likely split; it cannot certify that the model never saw test rows.
"""

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
model = joblib.load("churn_model.pkl")
train, test = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["Churn"]
)

features = list(model.feature_names_in_)
probabilities = model.predict_proba(test[features])[:, 1]
predictions = (probabilities >= 0.5).astype(int)
actual = test["Churn"]

print("Test rows:", len(test))
print("Confusion matrix [TN, FP, FN, TP]:",
      confusion_matrix(actual, predictions, labels=[0, 1]).ravel().tolist())
print("Churn precision:", round(precision_score(actual, predictions), 4))
print("Churn recall:", round(recall_score(actual, predictions), 4))
print("Churn F1:", round(f1_score(actual, predictions), 4))
print("ROC-AUC:", round(roc_auc_score(actual, probabilities), 4))
