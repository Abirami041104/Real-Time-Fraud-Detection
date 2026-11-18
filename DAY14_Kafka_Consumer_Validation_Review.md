# Day 14: Kafka Consumer Validation Review

## Objective
Review how incoming Kafka transactions are received and validated before fraud prediction.

## Existing implementation
- Kafka consumer subscribes to the transactions topic.
- JSON messages are deserialized into Python dictionaries.
- Required model features are extracted before prediction.
- Key, type and value errors are caught.
- Invalid transactions are skipped instead of terminating the processing loop.
- Processing errors are counted through Prometheus.

## Review findings
The consumer has basic exception handling. Explicit checks for missing transaction IDs, unsupported transaction types, negative amounts and non-finite numeric values should be added and tested.

## Conclusion
The consumer error-handling flow was reviewed and validation improvements were identified.
