# Day 8: Fraud Detection Threshold Optimization

## Objective
Evaluate different classification probability thresholds to improve fraud detection precision while maintaining high fraud recall.

## Work Completed
1. Loaded the existing trained Random Forest model.
2. Used a stratified sample of 500,000 transactions.
3. Created an 80:20 training and testing split.
4. Generated fraud prediction probabilities using `predict_proba()`.
5. Tested six classification thresholds: 0.1, 0.2, 0.3, 0.4, 0.5 and 0.6.
6. Compared precision, recall, F1-score, true positives, false positives and false negatives.
7. Exported the comparison results to a CSV report.

## Threshold Comparison

| Threshold | Precision | Recall | F1-score | False Positives | False Negatives |
|---|---:|---:|---:|---:|---:|
| 0.1 | 78.18% | 100% | 0.8776 | 36 | 0 |
| 0.2 | 90.85% | 100% | 0.9520 | 13 | 0 |
| 0.3 | 96.27% | 100% | 0.9810 | 5 | 0 |
| 0.4 | 97.73% | 100% | 0.9885 | 3 | 0 |
| 0.5 | 98.47% | 100% | 0.9923 | 2 | 0 |
| 0.6 | 99.23% | 100% | 0.9961 | 1 | 0 |

## Key Findings
- All six thresholds detected all 129 fraud transactions in the test set.
- Increasing the threshold reduced false-positive alerts.
- Threshold 0.6 achieved the highest precision (99.23%) among the tested values.
- At threshold 0.6, only one legitimate transaction was incorrectly classified as fraud.
- No false negatives were observed in this test.

## Selected Threshold
Threshold 0.6 produced the best observed balance of precision and recall in this experiment.

This is an experimental result, not a confirmed production threshold. A production threshold should also consider the financial cost of missed fraud, false alerts and investigation capacity.

## Output
- Script: `src/day8_threshold_optimization.py`
- Results: `reports/day8_threshold_comparison.csv`

## Conclusion
Completed probability threshold optimization for the fraud detection model. The experiment demonstrated how changing the classification threshold affects precision, recall and false-positive alerts without retraining the model.

Note: The experiment uses the synthetic PaySim dataset and a stratified random sample. Further validation on different data and chronological test sets is required before real-world deployment.