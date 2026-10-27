\# Day 3 — Exploratory Data Analysis (EDA)



\## Project

Real-Time Financial Fraud Detection Pipeline



\## Objective

Explore the PaySim transaction dataset to understand its structure, assess data quality, examine fraud distribution, and identify useful patterns for fraud detection.



\## 1. Dataset Overview

\- Total transactions: 6,362,620

\- Total columns: 11

\- Target variable: `isFraud`

\- Transaction types: CASH\_IN, CASH\_OUT, DEBIT, PAYMENT, TRANSFER



\## 2. Data Quality

\- Missing values: 0 across all columns

\- Exact duplicate rows: 0

\- Reviewed column names and data types.



\## 3. Fraud Distribution

\- Legitimate transactions: 6,354,407 (99.8709%)

\- Fraudulent transactions: 8,213 (0.1291%)



\### Key Finding

The dataset is highly imbalanced because fraudulent transactions represent only a small fraction of all transactions. This should be considered when evaluating machine learning models.



\## 4. Fraud by Transaction Type

\- CASH\_OUT: 4,116 fraudulent transactions

\- TRANSFER: 4,097 fraudulent transactions

\- PAYMENT: 0 fraudulent transactions in this dataset

\- CASH\_IN: 0 fraudulent transactions in this dataset

\- DEBIT: 0 fraudulent transactions in this dataset



Fraud cases in this dataset were observed in CASH\_OUT and TRANSFER transactions. This finding describes the analysed dataset and should not be treated as a universal rule for real-world banking.



\## 5. Transaction Amount Statistics

\- Mean amount: approximately 179,861.90

\- Median amount: approximately 74,871.94

\- Minimum amount: 0

\- Maximum amount: approximately 92,445,521.76



The mean is higher than the median, indicating that large transaction amounts influence the distribution.



\## 6. Reports Generated

The analysis script generated these CSV reports in the local project:

\- `reports/dataset\_summary.csv`

\- `reports/fraud\_distribution.csv`

\- `reports/fraud\_by\_type.csv`



\## 7. Day 3 Deliverables

\- Executed the existing Python dataset analysis script.

\- Examined dataset structure and data quality.

\- Analysed fraud distribution and transaction types.

\- Summarised transaction amount statistics.

\- Documented the findings for the project.



\## Next Step

Continue feature-level analysis and investigate transaction patterns that may help distinguish fraudulent from legitimate activity.

