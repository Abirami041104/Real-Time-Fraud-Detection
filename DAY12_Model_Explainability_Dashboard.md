\# DAY 12 – MODEL EXPLAINABILITY DASHBOARD INTEGRATION



\## Objective

Integrate the SHAP model explainability results generated on Day 11 into the existing Streamlit fraud detection dashboard.



\## Work Completed



1\. Integrated the saved SHAP feature importance results into the dashboard.

2\. Added a global feature importance visualization using Plotly.

3\. Integrated the SHAP summary plot.

4\. Integrated the SHAP importance bar plot.

5\. Added an individual fraud transaction waterfall explanation.

6\. Displayed the SHAP sample predictions in a dashboard table.

7\. Added readable feature labels for the global feature importance chart.

8\. Verified that all explainability sections render correctly in Streamlit.



\## Dashboard Sections Added



| Section | Description |

|---|---|

| Global Feature Importance | Displays the average absolute SHAP impact of each feature |

| SHAP Summary Plot | Shows feature impact distribution across sampled transactions |

| SHAP Importance Bar Plot | Visualizes overall feature importance |

| Individual Fraud Explanation | Explains the model output for one selected fraud transaction from the Day 11 analysis |

| SHAP Sample Predictions | Displays sampled transaction details and fraud probabilities |



\## Key Findings



\- Amount / Origin Balance is the most influential feature in the SHAP analysis.

\- Transaction Amount is the second most influential feature.

\- Transaction type and account balance features also contribute to model predictions.

\- The waterfall plot demonstrates how individual feature contributions combine to produce a fraud prediction.



\## Integration Details



\- Framework: Streamlit

\- Visualization: Plotly, Matplotlib

\- Explainability: SHAP

\- Source folder: `reports/day11\_shap/`

\- Main dashboard: `dashboard.py`



The dashboard reads the previously generated SHAP output files rather than recalculating SHAP explanations for every live transaction.



\## Validation



\- Python syntax validation completed successfully using `python -m py\_compile dashboard.py`.

\- Streamlit dashboard restarted successfully.

\- All five explainability sections were visually verified in the browser.

\- Existing dashboard functionality was retained.



\## Conclusion



Successfully integrated model explainability into the real-time fraud detection dashboard. The dashboard now provides both fraud monitoring information and visual explanations of the machine learning model's behaviour.



\## Day 12 Status



\*\*Completed – Model Explainability Dashboard Integration\*\*



