# Day 18: Spark Analytics Review

## Objective
Review the Spark components in the real-time fraud detection project.

## Existing project components
- spark_processor.py
- spark_live_analytics.py

## Architecture
Spark is used as the distributed data-processing and analytics component alongside the Kafka transaction pipeline.

## Review checklist
1. Confirm the Spark runtime starts successfully.
2. Confirm the input transaction data is accessible.
3. Verify the expected transaction columns.
4. Inspect fraud and transaction-type aggregations.
5. Check output records for missing or invalid values.
6. Record runtime errors for troubleshooting.

## Conclusion
The Spark analytics components and their role in the project architecture were documented. Final runtime verification should be performed before the review.
