# Retention analysis and action playbook

## What the sample shows

- 2,037 of 10,000 customers left (20.37%).
- Those customers held 24.26% of the balance recorded in the sample.
- This is an association in historical sample data. It does not establish why any customer left or how much revenue was lost.

## Suggested workflow

1. Score a customer batch in the Streamlit app. Review high probability accounts and their balances first.
2. Check the customer's recent service history and eligibility before outreach. Use the app's suggested action as a prompt for a human review.
3. Ask why the customer may leave. Record the reason, the action taken, and whether the customer stays.
4. Compare results with a suitable control group before claiming that an offer reduces churn.

## Illustrative risk tiers

| Tier | Model probability | Suggested next step |
| --- | --- | --- |
| High | 70% or above | Prioritize a human review and timely outreach |
| Medium | 40% to under 70% | Review engagement and ask for feedback |
| Low | Under 40% | Continue normal service and monitor changes |

The thresholds are illustrative. Revisit them after measuring calibration, outreach capacity, and observed outcomes. A score of probability × balance is a prioritization measure. It is neither expected revenue nor a guaranteed balance loss.

## What to measure next

- Precision among contacted customers and churn recall at the chosen outreach volume.
- Offer acceptance, actual retention after an agreed period, and cost per retained customer.
- Calibration of predicted probabilities and results by segment.

Avoid describing feature importance as a causal explanation. Recommendations need approval from the bank's product and compliance teams before real customer use.
