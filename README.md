# Bank Customer Churn Prediction & Retention Analytics

End-to-end churn analytics project: MySQL data assessment → feature engineering & Random Forest model in Python → Power BI dashboard → Streamlit prediction interface → retention playbook & business memo.

**Tools:** MySQL, Python (Pandas, Scikit-learn), Power BI, Google Sheets, Streamlit

---

## Key Findings

- Assessed **10,000 bank customer records** in MySQL, identifying anomalies such as churned customers holding **24% of total balance** despite being only **20% of headcount**.
- Engineered **10 customer features** and trained a **Random Forest classifier** achieving **74% F1-score** and **78% recall**, prioritizing recall to minimize missed at-risk customers.
- Designed a **Power BI dashboard** breaking down churn by **Geography, Age, and Tenure**, supporting data-driven decision-making.
- Compiled findings into a **retention playbook and business memo**, and built a **Streamlit prediction interface**, quantifying the business impact of high-value customer attrition.

---

## Repo Structure

```
├── sql/            # MySQL EDA queries — nulls, distributions, headcount vs. balance analysis
├── python/          # Feature engineering + Random Forest / Logistic Regression notebooks
├── powerbi/          # .pbix dashboard file + screenshots
├── streamlit/         # app.py — live churn prediction interface
├── docs/            # Retention playbook (PDF) + business memo (PDF)
└── data/            # Dataset (or source link, if not included raw)
```

## 1. MySQL — Data Assessment

- Verified data integrity across 10,000 records (zero nulls).
- Analyzed churn distribution across gender and geography.
- Core finding: churned customers are ~20% of headcount but hold ~24% of total balance — a disproportionate revenue risk.

See [`/sql`](./sql) for queries.

## 2. Python — Feature Engineering & Modeling

- Engineered 10 customer features (tenure buckets, balance-to-salary ratio, product engagement, etc.).
- Trained a Random Forest classifier with balanced class weights, benchmarked against Logistic Regression.
- **Result: 74% F1-score, 78% recall** — recall prioritized to minimize missed at-risk customers.

See [`/python`](./python) for notebooks.

## 3. Power BI — Dashboard

- Interactive dashboard breaking down churn by Geography, Age, and Tenure.
- Built for stakeholder-facing, data-driven retention decisions.

See [`/powerbi`](./powerbi) for the `.pbix` file and screenshots.

## 4. Streamlit — Prediction Interface

- Live interface for predicting churn risk on new customer inputs.
- [Live demo](#) *(add link once deployed)*

See [`/streamlit`](./streamlit) for `app.py`.

## 5. Business Deliverables

- **Retention Playbook** — actionable recommendations by customer segment.
- **Business Memo** — quantifies the revenue impact of high-value customer attrition.

See [`/docs`](./docs).

---

## Author

**Rishi Bhor** — [LinkedIn](https://linkedin.com/in/rishi-bhor-02a23b) · rishibhor326@gmail.com
