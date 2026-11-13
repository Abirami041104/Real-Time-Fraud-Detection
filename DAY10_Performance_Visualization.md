
# Day 10: Fraud Detection Performance Visualization

## Objective
Develop a professional interactive dashboard to evaluate fraud detection model performance across different classification thresholds.

## Work Completed

1. Loaded threshold comparison results from Day 8.
2. Created precision, recall and F1-score visualizations.
3. Compared false-positive and false-negative counts.
4. Developed a confusion matrix visualization.
5. Built an interactive Streamlit dashboard using Plotly.
6. Added dynamic threshold selection.
7. Displayed automated model performance insights.
8. Added a detailed threshold comparison table.

## Dashboard Features

- KPI cards for Precision, Recall, F1-score and False Positives.
- Interactive performance comparison chart.
- Classification error analysis.
- Interactive confusion matrix.
- Detailed threshold comparison table.
- Automated model insights.
- Dynamic threshold selection.

## Key Findings

Best observed threshold: 0.6

| Metric | Result |
|---|---:|
| Precision | 99.23% |
| Recall | 100% |
| F1-score | 0.9961 |
| True Positives | 129 |
| False Positives | 1 |
| False Negatives | 0 |
| True Negatives | 99,870 |

## Threshold Analysis

- Precision increased from 78.18% to 99.23%.
- Recall remained at 100% across all tested thresholds.
- False positives decreased from 36 to 1.
- False negatives remained at zero.
- Threshold 0.6 achieved the highest observed F1-score.

## Technologies Used

- Python
- Pandas
- Streamlit
- Plotly
- Matplotlib

## Output Files

- `src/day10_performance_visualization.py`
- `dashboards/day10_performance_dashboard.py`
- `reports/day10_visualizations/performance_metrics.png`
- `reports/day10_visualizations/classification_errors.png`
- `reports/day10_visualizations/f1_score_comparison.png`

## Conclusion

Successfully developed a professional interactive dashboard for fraud detection performance analysis.

The dashboard allows users to compare classification thresholds, understand model errors and evaluate performance through interactive charts and a confusion matrix.

Note: Results are based on a stratified random test sample from the synthetic PaySim dataset. Additional chronological validation and business-cost analysis are required before production deployment.
