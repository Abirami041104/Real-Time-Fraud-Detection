
# Day 9: Fraud Detection Model Error Analysis

## Objective
Identify and analyse incorrectly classified transactions using the trained Random Forest model and the selected probability threshold of 0.6.

## Work Completed
1. Loaded the processed PaySim dataset.
2. Created a stratified sample of 500,000 transactions.
3. Generated an 80:20 train-test split.
4. Loaded the saved Random Forest model.
5. Generated fraud prediction probabilities.
6. Applied the 0.6 classification threshold.
7. Identified false positives and false negatives.
8. Analysed misclassified transactions by transaction type.
9. Exported detailed error analysis reports.

## Confusion Matrix Results

| Metric | Count |
|---|---:|
| True Negatives | 99,870 |
| False Positives | 1 |
| False Negatives | 0 |
| True Positives | 129 |
| Total Test Transactions | 100,000 |
| Total Misclassified Transactions | 1 |

## Misclassified Transaction Analysis

One false-positive transaction was identified.

| Attribute | Value |
|---|---|
| Step | 332 |
| Transaction Type | CASH_OUT |
| Amount | 0.01 |
| Original Balance | 0.00 |
| Destination Balance | 262792.79 |
| Actual Class | Legitimate |
| Predicted Class | Fraud |
| Fraud Probability | 0.628692 |
| Error Type | FALSE_POSITIVE |

## Key Findings

- The model correctly classified 99,999 out of 100,000 test transactions.
- All 129 actual fraud transactions were detected.
- No false negatives were observed.
- Only one legitimate transaction was incorrectly flagged.
- The false-positive transaction was a CASH_OUT transaction with an amount of 0.01.
- Its fraud probability was approximately 62.87%, exceeding the selected threshold of 0.6.

## Output Files

- Script: `src/day9_error_analysis.py`
- Detailed errors: `reports/day9_misclassified_transactions.csv`
- Error summary: `reports/day9_error_summary.csv`

## Conclusion

Completed model error analysis using a classification threshold of 0.6. The experiment identified one false-positive transaction and no false negatives in the test sample.

The results provide insight into the model's classification errors and help identify transactions requiring further investigation.

Note: Results are based on the synthetic PaySim dataset and a stratified random test sample. Further validation is necessary before real-world deployment.
