import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import shap
import matplotlib.pyplot as plt

FEATURES = [
    'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
    'Num Of Products', 'Has Credit Card', 'Is Active Member', 'Estimated Salary'
]
ALIASES = {
    'NumOfProducts': 'Num Of Products',
    'HasCrCard': 'Has Credit Card',
    'IsActiveMember': 'Is Active Member',
    'EstimatedSalary': 'Estimated Salary'
}


def prepare_batch(df):
    """Validate the uploaded schema before calling the model."""
    for old, new in ALIASES.items():
        if old in df and new in df:
            raise ValueError(f"Both {old} and {new} are present; keep only one.")
    prepared = df.rename(columns=ALIASES)
    missing = [column for column in FEATURES if column not in prepared]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if prepared[FEATURES].isna().any().any():
        raise ValueError("Required model fields contain blank values.")
    if prepared.empty:
        raise ValueError("Upload a CSV with at least one customer.")
    return prepared[FEATURES]

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
        actions.append("Re-engagement: Review an eligible engagement offer")
        
    balance = row.get('Balance', 0.0)
    if balance >= 50000 and churn_prob >= 0.5:
        actions.append("Priority Outreach: Review for a relationship manager call")
        
    if not actions:
        actions.append("Targeted Retention: Request feedback and review available offers")
        
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
                if prob >= 0.7:
                    st.error("⚠️ Status: High Churn Risk")
                elif prob >= 0.4:
                    st.warning("Status: Medium Churn Risk")
                else:
                    st.success("✅ Status: Low Churn Risk")
            with col2:
                st.metric("Balance-weighted risk score", f"${capital_risk:,.2f}")
                st.caption("Probability × account balance; this is not a revenue or loss forecast.")

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
        try:
            df_batch = pd.read_csv(uploaded_file)
            df_batch_prep = prepare_batch(df_batch)
            df_batch = df_batch.rename(columns=ALIASES).copy()
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Cannot score this CSV: {exc}")
            st.stop()
        
        # Rename columns to ensure pipeline compatibility
        # Batch scoring & Risk Calculation
        churn_probs = model.predict_proba(df_batch_prep)[:, 1]
        df_batch['Churn_Probability'] = churn_probs
        df_batch['Balance_Weighted_Risk'] = df_batch['Churn_Probability'] * df_batch_prep['Balance'].to_numpy()
        df_batch['Capital_at_Risk'] = df_batch['Balance_Weighted_Risk']

        def assign_risk(p):
            return 'High Risk' if p >= 0.7 else ('Medium Risk' if p >= 0.4 else 'Low Risk')

        df_batch['Risk_Tier'] = df_batch['Churn_Probability'].apply(assign_risk)
        
        # Apply Automated Actions
        df_batch['Recommended_Action'] = [
            recommend_retention_action(row, p) for p, (_, row) in zip(churn_probs, df_batch_prep.iterrows())
        ]

        # Executive KPIs
        total_customers = len(df_batch)
        total_balance = df_batch_prep['Balance'].sum()
        total_risk = df_batch['Capital_at_Risk'].sum()
        high_risk_count = (df_batch['Risk_Tier'] == 'High Risk').sum()
        high_risk_bal = df_batch_prep.loc[df_batch['Risk_Tier'] == 'High Risk', 'Balance'].sum()
        st.session_state['scored_df'] = df_batch.copy()

        st.markdown("### Executive Risk Summary")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Balance-weighted risk score", f"${total_risk:,.2f}", f"{(total_risk/total_balance)*100:.1f}% of balances" if total_balance else "0%")
        st.caption("This score sums churn probability × balance. It does not estimate bank revenue or realized loss.")
        k2.metric("High Risk Accounts", f"{high_risk_count:,}", f"{(high_risk_count/total_customers)*100:.1f}% of total")
        k3.metric("High Risk Capital", f"${high_risk_bal:,.2f}")
        k4.metric("Avg Churn Risk", f"{df_batch['Churn_Probability'].mean()*100:.1f}%")

        # Dataframe Display & Export
        st.markdown("---")
        st.subheader("Scored Customer Portfolio")
        tier_filter = st.multiselect("Filter Risk Tiers", ['High Risk', 'Medium Risk', 'Low Risk'], default=['High Risk', 'Medium Risk'])
        filtered_df = df_batch[df_batch['Risk_Tier'].isin(tier_filter)].sort_values(by='Balance_Weighted_Risk', ascending=False)

        display_cols = [c for c in ['CustomerId', 'Surname', 'Geography', 'Balance', 'Churn_Probability', 'Balance_Weighted_Risk', 'Risk_Tier', 'Recommended_Action'] if c in filtered_df.columns]
        st.dataframe(filtered_df[display_cols], use_container_width=True)

        csv_data = df_batch.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Scored Financial Risk Report", csv_data, "bank_churn_financial_risk_report.csv", "text/csv")

# --- TAB 3: EXECUTIVE MACRO ANALYTICS ---
with tab3:
    st.header("📈 Executive Portfolio & Cohort Analysis")
    st.write("Portfolio breakdown across geography, tenure, and product count.")

    scored_df = st.session_state.get('scored_df')
    if scored_df is None:
        st.info("Upload a customer CSV in Batch Risk Analysis to view portfolio charts.")
    else:
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Balance-weighted risk by Geography")
            fig_geo = px.bar(
                scored_df.groupby('Geography')['Capital_at_Risk'].sum().reset_index(),
                x='Geography',
                y='Capital_at_Risk',
                color='Geography',
                title="Balance-weighted risk score by country",
                labels={'Capital_at_Risk': 'Balance-weighted risk score ($)'}
            )
            st.plotly_chart(fig_geo, use_container_width=True)

        with c2:
            st.subheader("Churn Risk by Product Count")
            fig_prod = px.box(
                scored_df,
                x='Num Of Products',
                y='Churn_Probability',
                color='Num Of Products',
                title="Churn Probability Distribution across Product Holdings",
                labels={'Num Of Products': 'Number of Products', 'Churn_Probability': 'Churn Prob'}
            )
            st.plotly_chart(fig_prod, use_container_width=True)

        st.markdown("---")
        st.subheader("Tenure Decay & Risk Concentration")
        
        fig_tenure = px.line(
            scored_df.groupby('Tenure')['Capital_at_Risk'].sum().reset_index(),
            x='Tenure',
            y='Capital_at_Risk',
            markers=True,
            title="Balance-weighted risk score by customer tenure",
            labels={'Tenure': 'Tenure (Years)', 'Capital_at_Risk': 'Balance-weighted risk score ($)'}
        )
        st.plotly_chart(fig_tenure, use_container_width=True)
