# Day 16: MongoDB Fraud Alert Storage Review

## Objective
Review persistence of model-identified fraud alerts.

## Existing implementation
- Connects to the local MongoDB server.
- Uses the fraud_detection database.
- Uses the fraud_alerts collection.
- Stores transactions predicted as fraudulent.
- Saves transaction ID, timestamp, transaction type, amount, balances, prediction and fraud probability.
- Uses update_one with upsert=True and transaction_id as the matching key.

## Review findings
The upsert operation helps avoid multiple alert documents for the same transaction ID.

The transaction ID must be unique and stable. Reusing IDs for different transactions could overwrite an existing alert.

## Conclusion
The MongoDB alert-storage workflow and its duplicate-handling approach were documented.
