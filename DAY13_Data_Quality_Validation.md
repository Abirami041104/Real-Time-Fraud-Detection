\# Day 13: Live Transaction Data Quality Validation



\## Objective

Develop an automated Python validation script to assess the quality and consistency of live fraud prediction records.



\## Implementation

Created `src/day13\_data\_quality.py` using Pandas.



\### Validation checks

1\. Required column validation.

2\. Missing value detection.

3\. Repeated transaction ID detection.

4\. Conflicting transaction detail detection.

5\. Prediction label validation (0 or 1).

6\. Fraud probability range validation (0–1).

7\. Prediction and status consistency validation.



\## Results



| Metric | Result |

|---|---:|

| Total records | 220 |

| Unique transaction IDs | 20 |

| Fraud predictions | 11 |

| Legitimate predictions | 209 |

| Repeated ID records | 200 |

| Conflicting transaction IDs | 0 |

| Exact duplicate rows | 0 |



\## Findings

All validation checks passed. The script identified 200 repeated transaction ID records, but no conflicting transaction details.



The repeated records appear to represent repeated test-run data. They are reported as a warning rather than treated as confirmed data corruption.



\## Conclusion

Automated data-quality validation was implemented successfully. The script helps identify missing data, invalid predictions, inconsistent statuses, and repeated or conflicting transaction records before further analysis.

