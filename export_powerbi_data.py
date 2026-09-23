import pandas as pd
import joblib

# 1. Load data and model
df = pd.read_csv('Churn_Modelling.csv')
model = joblib.load('churn_model.pkl')

# 2. Rename columns to match model expected feature names
df_prep = df.rename(columns={
    'NumOfProducts': 'Num Of Products',
    'HasCrCard': 'Has Credit Card',
    'IsActiveMember': 'Is Active Member',
    'EstimatedSalary': 'Estimated Salary'
})

# 3. Calculate predictions and risk metrics
churn_probs = model.predict_proba(df_prep)[:, 1]
df['Churn_Probability'] = churn_probs
df['Capital_at_Risk'] = df['Churn_Probability'] * df['Balance']

# 4. Assign Risk Tiers
def assign_risk(p):
    return 'High Risk' if p >= 0.7 else ('Medium Risk' if p >= 0.4 else 'Low Risk')

df['Risk_Tier'] = df['Churn_Probability'].apply(assign_risk)

# 5. Export for Power BI
df.to_csv('bank_churn_financial_risk_report.csv', index=False)
print("✅ Success: 'bank_churn_financial_risk_report.csv' created successfully for Power BI!")