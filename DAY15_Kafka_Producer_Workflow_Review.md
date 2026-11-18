# Day 15: Kafka Producer Workflow Review

## Objective
Document the transaction streaming workflow.

## Existing implementation
- Reads processed_transactions.csv.
- Reads a sample of 20 transactions.
- Generates transaction IDs in the TXN_0001 format.
- Converts transaction fields into numeric values.
- Serializes transaction dictionaries into JSON.
- Publishes messages to the transactions Kafka topic.
- Waits for message delivery and flushes the producer.

## Transaction fields
transaction_id, step, type, amount, oldbalanceOrg, oldbalanceDest, amount_to_origin_balance and isFraud.

## Review finding
The original isFraud field is dataset information and must not be used as an input to the prediction model.

## Conclusion
The producer-to-Kafka message workflow was documented for project review.
