import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import shap
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Bank Customer Churn Predictor", layout="wide")
st.title("🏦 Bank Customer Churn & Risk Analytics Engine")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('churn_model.pkl')
        return model
    except Exception as e:
        st.error(f"Could not load churn_model.pkl. Error: {e}")
        return None

model = load_model()

# --- RETENTION ACTION ENGINE ---
def recommend_retention_action(row, churn_prob):
    """Maps customer features and risk scores to specific retention playbooks."""
    if churn_prob < 0.4:
        return "Standard Nurture: Periodic Relationship Touchpoint"
    
    actions = []
    
    num_products = row.get('Num Of Products', row.get('NumOfProducts', 1))
    if num_products == 1:
        actions.append("Cross-Sell: Offer No-Fee Cashback Credit Card or Secondary Account")
        
    is_active = row.get('Is Active Member', row.get('IsActiveMember', 1))
    if is_active == 0:
        actions.append("Re-engagement: Send 0.5% Bonus Interest High-Yield Savings Promo")
        
    balance = row.get('Balance', 0.0)
    if balance >= 50000 and churn_prob >= 0.5:
        actions.append("VIP Outreach: Assign Dedicated Relationship Manager Call within 24 Hours")
        
    if not actions:
        actions.append("Targeted Retention: Send Custom Fee-Waiver & Feedback Survey")
        
    return " | ".join(actions)

# --- SIDEBAR EXECUTIVE LINK ---
with st.sidebar:
    st.header("🔗 Executive Analytics")
    st.markdown("Use this Streamlit app for real-time customer scoring, SHAP analysis, and macro cohort trends natively.")
    st.markdown("---")

# --- TABS SETUP ---
tab1, tab2, tab3 = st.tabs([
    "👤 Single Customer Evaluator", 
    "📊 Batch Risk Analysis", 
    "📈 Executive Macro Analytics"
])

# --- TAB 1: SINGLE CUSTOMER EVALUATOR ---
with tab1:
    st.header("Single Customer Prediction")
    st.write("Input customer details to calculate real-time churn risk & expected financial impact.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        credit_score = st.number_input("Credit Score", 300, 850, 650)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 18, 100, 40)

    with col_b:
        tenure = st.number_input("Tenure (Years)", 0, 10, 5)
        balance = st.number_input("Balance ($)", 0.0, value=50000.0, step=1000.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)

    with col_c:
        has_crcard = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        estimated_salary = st.number_input("Estimated Salary ($)", 0.0, value=75000.0, step=1000.0)

    if st.button("Calculate Churn Risk", type="primary"):
        if model is not None:
            # Construct DataFrame with exact column names expected by the model
            input_df = pd.DataFrame([{
                'CreditScore': credit_score,
                'Geography': geography,
                'Gender': gender,
                'Age': age,
                'Tenure': tenure,
                'Balance': balance,
                'Num Of Products': num_products,
                'Has Credit Card': has_crcard,
                'Is Active Member': is_active,
                'Estimated Salary': estimated_salary
            }])

            # Generate Prediction & Risk
            prob = model.predict_proba(input_df)[0][1]
            capital_risk = prob * balance
            recommended_action = recommend_retention_action(input_df.iloc[0], prob)

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Churn Probability", f"{prob*100:.1f}%")
                if prob >= 0.5:
                    st.error("⚠️ Status: High Churn Risk")
                else:
                    st.success("✅ Status: Low Churn Risk")
            with col2:
                st.metric("Expected Balance at Risk", f"${capital_risk:,.2f}")

            st.subheader("💡 Recommended Retention Playbook")
            st.info(f"**Action Plan:** {recommended_action}")

            # SHAP Explainability
            st.markdown("---")
            st.subheader("🧠 Model Decision Drivers (SHAP Explanation)")
            st.write("This plot breaks down how each specific feature pushed the prediction higher (red) or lower (blue).")
            
            try:
                if hasattr(model, 'named_steps'):
                    preprocessor = model.steps[0][1]
                    classifier = model.steps[-1][1]
                    
                    # Transform the input to get feature values
                    X_transformed = preprocessor.transform(input_df)
                    
                    # Retrieve feature names for the plot
                    if hasattr(preprocessor, 'get_feature_names_out'):
                        feature_names = preprocessor.get_feature_names_out()
                    else:
                        feature_names = [f"Feature {i}" for i in range(X_transformed.shape[1])]
                        
                    # Calculate SHAP values
                    explainer = shap.TreeExplainer(classifier)
                    shap_values = explainer(X_transformed)
                    
                    # Render the Waterfall Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.plots.waterfall(
                        shap.Explanation(
                            values=shap_values.values[0, :, 1] if len(shap_values.values.shape) == 3 else shap_values.values[0], 
                            base_values=shap_values.base_values[0, 1] if isinstance(shap_values.base_values[0], (list, np.ndarray)) else shap_values.base_values[0], 
                            data=X_transformed[0], 
                            feature_names=feature_names
                        ),
                        show=False
                    )
                    st.pyplot(fig)
                else:
                    st.warning("SHAP explanation requires the model to be loaded as a Scikit-Learn Pipeline.")
            except Exception as e:
                st.error(f"Could not generate SHAP plot: {e}")

# --- TAB 2: BATCH RISK & FINANCIAL IMPACT ---
with tab2:
    st.header("📊 Batch Risk & Financial Impact Analysis")
    uploaded_file = st.file_uploader("Upload Customer Batch CSV", type=["csv"])

    if uploaded_file is not None and model is not None:
        df_batch = pd.read_csv(uploaded_file)
        
        # Rename columns to ensure pipeline compatibility
        column_mapping = {
            'NumOfProducts': 'Num Of Products',
            'HasCrCard': 'Has Credit Card',
            'IsActiveMember': 'Is Active Member',
            'EstimatedSalary': 'Estimated Salary'
        }
        df_batch_prep = df_batch.rename(columns=column_mapping)

        # Batch scoring & Risk Calculation
        churn_probs = model.predict_proba(df_batch_prep)[:, 1]
        df_batch['Churn_Probability'] = churn_probs
        df_batch['Capital_at_Risk'] = df_batch['Churn_Probability'] * df_batch['Balance']

        def assign_risk(p):
            return 'High Risk' if p >= 0.7 else ('Medium Risk' if p >= 0.4 else 'Low Risk')

        df_batch['Risk_Tier'] = df_batch['Churn_Probability'].apply(assign_risk)
        
        # Apply Automated Actions
        df_batch['Recommended_Action'] = [
            recommend_retention_action(row, p) for p, (_, row) in zip(churn_probs, df_batch.iterrows())
        ]

        # Executive KPIs
        total_customers = len(df_batch)
        total_balance = df_batch['Balance'].sum()
        total_risk = df_batch['Capital_at_Risk'].sum()
        high_risk_count = (df_batch['Risk_Tier'] == 'High Risk').sum()
        high_risk_bal = df_batch[df_batch['Risk_Tier'] == 'High Risk']['Balance'].sum()

        st.markdown("### Executive Risk Summary")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Capital at Risk", f"${total_risk:,.2f}", f"{(total_risk/total_balance)*100:.1f}% of deposits" if total_balance else "0%")
        k2.metric("High Risk Accounts", f"{high_risk_count:,}", f"{(high_risk_count/total_customers)*100:.1f}% of total")
        k3.metric("High Risk Capital", f"${high_risk_bal:,.2f}")
        k4.metric("Avg Churn Risk", f"{df_batch['Churn_Probability'].mean()*100:.1f}%")

        # Dataframe Display & Export
        st.markdown("---")
        st.subheader("Scored Customer Portfolio")
        tier_filter = st.multiselect("Filter Risk Tiers", ['High Risk', 'Medium Risk', 'Low Risk'], default=['High Risk', 'Medium Risk'])
        filtered_df = df_batch[df_batch['Risk_Tier'].isin(tier_filter)].sort_values(by='Capital_at_Risk', ascending=False)

        display_cols = [c for c in ['CustomerId', 'Surname', 'Geography', 'Balance', 'Churn_Probability', 'Capital_at_Risk', 'Risk_Tier', 'Recommended_Action'] if c in filtered_df.columns]
        st.dataframe(filtered_df[display_cols], use_container_width=True)

        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Scored Financial Risk Report", csv_data, "bank_churn_financial_risk_report.csv", "text/csv")

# --- TAB 3: EXECUTIVE MACRO ANALYTICS ---
with tab3:
    st.header("📈 Executive Portfolio & Cohort Analysis")
    st.write("Macro-level portfolio breakdown across geography, tenure, age, and product density.")

    try:
        # Load the batch report previously generated to drive macro visuals
        scored_df = pd.read_csv('bank_churn_financial_risk_report.csv')
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Capital at Risk by Geography")
            fig_geo = px.bar(
                scored_df.groupby('Geography')['Capital_at_Risk'].sum().reset_index(),
                x='Geography',
                y='Capital_at_Risk',
                color='Geography',
                title="Total Deposit Capital at Risk by Country",
                labels={'Capital_at_Risk': 'Capital at Risk ($)'}
            )
            st.plotly_chart(fig_geo, use_container_width=True)

        with c2:
            st.subheader("Churn Risk by Product Count")
            fig_prod = px.box(
                scored_df,
                x='NumOfProducts',
                y='Churn_Probability',
                color='NumOfProducts',
                title="Churn Probability Distribution across Product Holdings",
                labels={'NumOfProducts': 'Number of Products', 'Churn_Probability': 'Churn Prob'}
            )
            st.plotly_chart(fig_prod, use_container_width=True)

        st.markdown("---")
        st.subheader("Tenure Decay & Risk Concentration")
        
        fig_tenure = px.line(
            scored_df.groupby('Tenure')['Capital_at_Risk'].sum().reset_index(),
            x='Tenure',
            y='Capital_at_Risk',
            markers=True,
            title="Capital at Risk over Customer Tenure (Years)",
            labels={'Tenure': 'Tenure (Years)', 'Capital_at_Risk': 'Capital at Risk ($)'}
        )
        st.plotly_chart(fig_tenure, use_container_width=True)

    except Exception:
        st.info("Upload a batch file in Tab 2 and save it as `bank_churn_financial_risk_report.csv` in your project folder to view macro analytics.")