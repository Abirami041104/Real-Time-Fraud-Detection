# Day 5: Feature Validation and Model Evaluation

## Objective
Evaluate fraud detection performance and investigate the impact of removing balance-related features.

## Experiment 1: Existing Saved Model
The previously saved model was evaluated on a stratified random test sample of 100,000 transactions.

- Legitimate transactions correctly classified: 99,869
- Legitimate transactions incorrectly flagged: 2
- Fraud transactions correctly detected: 129
- Fraud transactions missed: 0
- Average Precision (PR-AUC): 1.000000
- ROC-AUC: 1.000000

These results apply to the existing random-split evaluation and should not be interpreted as proof of real-world performance.

## Experiment 2: Model Without Balance-Related Features
A separate Random Forest model was trained using a time-based split and only these features:

- step
- type
- amount

### Evaluation Results

| Metric | Result |
|---|---:|
| Test transactions | 1,248,736 |
| Actual fraud transactions | 4,250 |
| Fraud detected | 1,244 |
| Fraud missed | 3,006 |
| Legitimate transactions incorrectly flagged | 11,022 |
| Fraud precision | 10% |
| Fraud recall | 29% |
| PR-AUC | 0.223544 |
| ROC-AUC | 0.816550 |

### Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 1,233,464 | 11,022 |
| Actual Fraud | 3,006 | 1,244 |

## Key Findings

1. The model without balance-related features detected 1,244 of 4,250 fraud transactions.
2. It missed 3,006 fraud transactions, demonstrating a limitation of this feature set.
3. It incorrectly flagged 11,022 legitimate transactions.
4. The saved model's random-split results and this experiment's time-based results use different evaluation setups and must not be compared as if they were identical tests.
5. Further validation is needed to understand the contribution of individual balance features and the amount-to-origin-balance ratio.

## Limitations

The PaySim dataset is simulated financial transaction data. Strong performance on this dataset does not guarantee equivalent performance on real banking transactions. Potential feature leakage and time-based generalization must be considered.

## Conclusion

Day 5 established a baseline evaluation and tested a reduced feature set. The experiment demonstrates that using only transaction step, type, and amount has limitations for identifying fraud in the evaluated time-based test set.

## Next Step

Investigate individual feature contributions and validate model performance using consistent evaluation splits.
