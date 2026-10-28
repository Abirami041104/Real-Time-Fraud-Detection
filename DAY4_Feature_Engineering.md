\# Day 4 — Feature Engineering and Fraud Pattern Analysis



\## Project

Real-Time Financial Fraud Detection Pipeline



\## Objective

Create additional transaction features and analyse how sender balance differences, transaction amounts, and transaction types relate to fraudulent activity.



\## 1. Dataset Analysed

\- Total transactions: 6,362,620

\- Total columns in processed dataset: 7

\- Fraudulent transactions: 8,213

\- Legitimate transactions: 6,354,407



\## 2. New Analytical Features



Created three additional features in the analysis script:



\- `origin\_balance\_difference`: Sender's original balance minus the transaction amount.

\- `absolute\_balance\_difference`: Absolute value of the difference between the sender's original balance and transaction amount.

\- `relative\_balance\_difference`: Absolute balance difference divided by the transaction amount plus one.



These features were created for analysis without modifying the original processed dataset.



\## 3. Balance Difference Analysis



Transactions were grouped into ranges according to their absolute balance difference.



| Absolute balance difference | Transactions | Fraud cases |

|---|---:|---:|

| 0 | 8,034 | 8,034 |

| 1–100 | 10,719 | 0 |

| 101–1,000 | 93,669 | 0 |

| 1,001–10,000 | 804,355 | 0 |

| 10,001–100,000 | 2,106,439 | 7 |

| Above 100,000 | 3,339,404 | 172 |



\### Key Finding

All 8,034 transactions in the zero-difference group were fraudulent in this dataset. However, 179 additional fraudulent transactions occurred in groups with nonzero differences.



This pattern may be useful for identifying suspicious transactions, but it should be validated before being used as a standalone fraud detection rule.



\## 4. Fraud Analysis by Transaction Type



\- CASH\_OUT: 2,237,500 transactions, including 4,116 fraud cases.

\- TRANSFER: 532,909 transactions, including 4,097 fraud cases.

\- CASH\_IN: 1,399,284 transactions, with no fraud cases in this dataset.

\- PAYMENT: 2,151,495 transactions, with no fraud cases in this dataset.

\- DEBIT: 41,432 transactions, with no fraud cases in this dataset.



TRANSFER had the highest observed fraud rate among transaction types, at approximately 0.7688%.



These observations describe the analysed PaySim dataset and are not universal rules for real-world financial transactions.



\## 5. Fraudulent vs Legitimate Transactions



| Metric | Legitimate | Fraudulent |

|---|---:|---:|

| Transaction count | 6,354,407 | 8,213 |

| Average transaction amount | 178,197 | 1,467,967 |

| Median transaction amount | 74,684.72 | 441,423.44 |

| Average absolute balance difference | 933,040.44 | 184,142.90 |



Fraudulent transactions had higher average and median amounts in this dataset, while their average absolute balance difference was lower.



\## 6. Reports Generated



The analysis script generated these CSV reports:



\- `reports/day4\_balance\_difference\_analysis.csv`

\- `reports/day4\_fraud\_by\_type\_analysis.csv`

\- `reports/day4\_feature\_comparison.csv`



\## 7. Day 4 Deliverables



\- Created `src/day4\_feature\_analysis.py`.

\- Engineered three additional analytical features.

\- Analysed fraud counts across balance-difference ranges.

\- Compared fraud patterns across transaction types.

\- Compared transaction amounts and balance differences for fraudulent and legitimate transactions.

\- Generated three CSV reports.



\## Next Step

Continue with feature validation and investigate how these analytical features could contribute to a reliable fraud detection model.



