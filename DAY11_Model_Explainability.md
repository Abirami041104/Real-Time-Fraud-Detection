# DAY 11 – Model Explainability Using SHAP

## 1. Objective
To understand how the trained Random Forest fraud detection model makes predictions using SHAP (SHapley Additive exPlanations).

## 2. Work Completed
- Installed and verified SHAP 0.51.0.
- Loaded the previously trained Random Forest pipeline.
- Selected 1,000 transactions from the processed PaySim dataset.
- Applied the existing preprocessing pipeline.
- Calculated SHAP values using TreeExplainer.
- Generated global feature importance and individual prediction explanations.

## 3. Dataset Details
- Dataset: PaySim processed transactions
- Sample size: 1,000 transactions
- Fraud cases: 4
- Transformed feature count: 10

## 4. SHAP Feature Importance

| Feature | Mean Absolute SHAP |
|---|---:|
| amount_to_origin_balance | 0.295974 |
| amount | 0.051534 |
| type_PAYMENT | 0.046367 |
| oldbalanceOrg | 0.044728 |
| type_CASH_IN | 0.036271 |
| oldbalanceDest | 0.026666 |
| type_CASH_OUT | 0.025365 |
| type_TRANSFER | 0.022546 |
| step | 0.012536 |
| type_DEBIT | 0.000126 |

## 5. Key Findings
1. amount_to_origin_balance was the most influential feature in the analysed sample.
2. Transaction amount was the second most influential feature.
3. Transaction type also contributed to model predictions.
4. The DEBIT encoded feature showed very low average SHAP impact.
5. SHAP helped explain both overall feature importance and individual predictions.

## 6. Individual Fraud Transaction Explanation

An example fraud transaction was analysed:

- Transaction type: CASH_OUT
- Amount: 93,483.23
- Origin balance: 93,483.23
- Destination balance: 86,882.30
- Amount-to-origin-balance ratio: 0.999989
- Step: 683

The waterfall plot showed:
- Base model output: approximately 0.50
- Final model output: approximately 0.98
- The amount-to-origin-balance feature made the strongest positive contribution.
- Destination balance made a negative contribution.

## 7. Visualizations Generated
1. shap_summary.png – SHAP summary plot.
2. shap_bar.png – Global feature importance.
3. fraud_waterfall.png – Individual fraud transaction explanation.

## 8. Additional Output Files
- feature_importance.csv
- sample_predictions.csv

## 9. Technologies Used
- Python
- SHAP 0.51.0
- Pandas
- Joblib
- Matplotlib
- Scikit-learn
- Random Forest

## 10. Conclusion
SHAP explainability analysis provided insights into how the Random Forest model makes fraud predictions. The amount-to-origin-balance feature had the highest average contribution magnitude in the analysed sample.

The analysis improves model interpretability by showing global feature contributions and individual transaction explanations.

**Limitation:** The analysis used a random sample of 1,000 transactions containing only four fraud cases. These findings are sample-specific and should not be treated as conclusive evidence of model behaviour across the entire dataset. SHAP explains model behaviour, not actual fraud causation.