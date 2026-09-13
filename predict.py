import pandas as pd
import joblib

# 1. Load trained pipeline
model = joblib.load('churn_model.pkl')

# 2. Define sample customer data with exact matching column names
sample_customers = pd.DataFrame([
    {
        'CreditScore': 600,
        'Geography': 'France',
        'Gender': 'Male',
        'Age': 32,
        'Tenure': 5,
        'Balance': 45000.0,
        'Num Of Products': 2,
        'Has Credit Card': 1,
        'Is Active Member': 1,
        'Estimated Salary': 50000.0,
    },
    {
        'CreditScore': 520,
        'Geography': 'Germany',
        'Gender': 'Female',
        'Age': 54,
        'Tenure': 2,
        'Balance': 125000.0,
        'Num Of Products': 1,
        'Has Credit Card': 1,
        'Is Active Member': 0,
        'Estimated Salary': 110000.0,
    },
])

# 3. Predict churn probabilities and labels
probabilities = model.predict_proba(sample_customers)[:, 1]
predictions = model.predict(sample_customers)

print("=== Inference Results ===")
for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
    status = 'Churn Risk' if pred == 1 else 'Loyal'
    print(f'Customer {i+1}: {status} | Churn Probability: {prob:.2%}')