import streamlit as st
import pandas as pd
import joblib

model = joblib.load('churn_model.pkl')

st.title('🏦 Bank Customer Churn Predictor')
st.write('Adjust customer parameters to evaluate churn probability in real time.')

col1, col2 = st.columns(2)

with col1:
    age = st.slider('Age', 18, 92, 38)
    balance = st.number_input('Account Balance ($)', value=60000.0)
    num_products = st.selectbox('Number of Products', [1, 2, 3, 4], index=0)
    is_active = st.selectbox('Is Active Member?', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
    geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])

with col2:
    credit_score = st.slider('Credit Score', 300, 850, 650)
    gender = st.selectbox('Gender', ['Female', 'Male'])
    tenure = st.slider('Tenure (Years)', 0, 10, 5)
    has_card = st.selectbox('Has Credit Card?', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
    salary = st.number_input('Estimated Salary ($)', value=75000.0)

input_data = pd.DataFrame([{
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'Num Of Products': num_products,
    'Has Credit Card': has_card,
    'Is Active Member': is_active,
    'Estimated Salary': salary
}])

if st.button('Predict Churn Risk'):
    prob = model.predict_proba(input_data)[0][1]
    st.subheader(f'Churn Probability: {prob:.1%}')
    if prob > 0.5:
        st.error('⚠️ High Churn Risk Customer')
    else:
        st.success('✅ Low Churn Risk Customer')
