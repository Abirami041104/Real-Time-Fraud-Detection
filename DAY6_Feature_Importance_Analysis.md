# Day 6: Feature Importance and Ablation Analysis

## Objective
Evaluate the contribution of individual feature groups to the Real-Time Financial Fraud Detection model using feature-ablation experiments.

## Dataset and Methodology
- Dataset: PaySim financial transaction dataset
- Total records: 6,362,620
- Training records: 5,113,884
- Testing records: 1,248,736
- Split method: Chronological split
- Time cutoff: Step 355
- Algorithm: Random Forest Classifier
- Evaluation metrics: PR-AUC and ROC-AUC

Three experiments were conducted using the same chronological split and model configuration.

## Experiments

### 1. All Features
Features used:
- step
- type
- amount
- oldbalanceOrg
- oldbalanceDest
- amount_to_origin_balance

### 2. Without Ratio Feature
The `amount_to_origin_balance` feature was removed to evaluate its contribution.

### 3. Without Balance Features
All balance-related features were removed, leaving:
- step
- type
- amount

## Results

| Experiment | Feature Count | PR-AUC | ROC-AUC |
|---|---:|---:|---:|
| All Features | 6 | 0.999997 | 1.000000 |
| Without Ratio | 5 | 0.593303 | 0.992886 |
| Without Balance Features | 3 | 0.240567 | 0.810207 |

## Key Findings

1. The model using all six features achieved a PR-AUC of approximately 1.0.
2. Removing `amount_to_origin_balance` reduced PR-AUC significantly to 0.593303.
3. Removing all balance-related features reduced PR-AUC further to 0.240567.
4. Balance-related information plays a major role in identifying fraudulent transactions in this dataset.
5. The experiments demonstrate the importance of evaluating feature contributions instead of relying only on overall model performance.

## Important Limitation

PaySim is a synthetic financial transaction dataset. Its simulated fraud patterns may make certain features unusually predictive. Therefore, these results should not be interpreted as proof of equivalent performance on real-world banking transactions.

## Output File
`reports/day6_feature_ablation.csv`

## Conclusion
Day 6 successfully evaluated the contribution of transaction and balance-related features through controlled feature-ablation experiments. The results highlight the importance of balance-related features for fraud detection on the PaySim dataset.

**Status: Completed**