import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load local CSV file & cleanup
df = pd.read_csv('Churn_Modelling.csv')
df.columns = df.columns.str.strip()

# Dynamic target detection
target_col = next((col for col in df.columns if col.lower() in ['exited', 'churn']), None)
drop_candidates = ['rownumber', 'customerid', 'surname', 'id', 'customer_id', 'row_number', target_col.lower()]
cols_to_drop = [c for c in df.columns if c.lower() in drop_candidates]

X = df.drop(columns=cols_to_drop)
y = df[target_col]

# 2. Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Identify Feature Types (Explicitly pass 'str' to fix Pandas warning)
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category', 'str']).columns.tolist()

# 4. Build Preprocessor & Model Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
    ]
)

model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# 5. Train Model
print("Training Random Forest Classifier...")
model_pipeline.fit(X_train, y_train)

# 6. Evaluate Performance
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# 7. Save Model Pipeline
joblib.dump(model_pipeline, 'churn_model.pkl')
print("\nModel saved successfully as 'churn_model.pkl'")