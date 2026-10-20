
import os
import csv
import json
import time
import joblib
import pandas as pd

from datetime import datetime
from kafka import KafkaConsumer
from pymongo import MongoClient
from prometheus_client import Counter, Histogram, start_http_server

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

MODEL_PATH = "models/fraud_detection_model.pkl"
LOG_FILE = "data/live_predictions.csv"

MONGODB_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "fraud_detection"
COLLECTION_NAME = "fraud_alerts"

FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_to_origin_balance",
]

LOG_COLUMNS = [
    "timestamp",
    "transaction_id",
    "type",
    "amount",
    "prediction",
    "fraud_probability",
    "status",
]

# --------------------------------------------------
# 2. PROMETHEUS METRICS
# --------------------------------------------------

TRANSACTIONS_PREDICTED = Counter(
    "fraud_detection_predictions_total",
    "Total transactions successfully predicted by the model",
    ["transaction_type", "prediction"],
)

FRAUD_ALERTS = Counter(
    "fraud_detection_fraud_alerts_total",
    "Total transactions predicted as fraudulent",
    ["transaction_type"],
)

PROCESSING_ERRORS = Counter(
    "fraud_detection_processing_errors_total",
    "Transaction processing errors",
    ["error_type"],
)

PROCESSING_TIME = Histogram(
    "fraud_detection_transaction_processing_seconds",
    "Time taken to process each transaction",
)

# --------------------------------------------------
# 3. PREPARE CSV LOG FILE
# --------------------------------------------------

os.makedirs("data", exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_COLUMNS)
        writer.writeheader()

# --------------------------------------------------
# 4. LOAD TRAINED MODEL
# --------------------------------------------------

print("Loading fraud detection model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# --------------------------------------------------
# 5. CONNECT TO MONGODB
# --------------------------------------------------

mongo_client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=3000,
)

mongo_client.admin.command("ping")

db = mongo_client[DATABASE_NAME]
fraud_alerts = db[COLLECTION_NAME]

print("MongoDB connected successfully!")
print(f"Fraud alert collection: {DATABASE_NAME}.{COLLECTION_NAME}")

# --------------------------------------------------
# 6. CONNECT TO KAFKA
# --------------------------------------------------

consumer = KafkaConsumer(
    "transactions",
    bootstrap_servers=["localhost:9092"],
    value_deserializer=lambda message: json.loads(
        message.decode("utf-8")
    ),
    auto_offset_reset="latest",
    enable_auto_commit=True,
    group_id="fraud-detection-group",
)

# --------------------------------------------------
# 7. START PROMETHEUS METRICS SERVER
# --------------------------------------------------

start_http_server(8000)

print("=" * 60)
print(" REAL-TIME FINANCIAL FRAUD DETECTION")
print(" Connected to Kafka: transactions")
print(" Connected to MongoDB: fraud_alerts")
print(" Prometheus metrics: http://localhost:8000/metrics")
print(" Waiting for incoming transactions...")
print("=" * 60)

# --------------------------------------------------
# 8. PROCESS INCOMING TRANSACTIONS
# --------------------------------------------------

try:
    for message in consumer:
        transaction = message.value
        processing_start = time.perf_counter()

        try:
            row = {
                feature: transaction[feature]
                for feature in FEATURES
            }

            data = pd.DataFrame([row], columns=FEATURES)

            prediction = int(model.predict(data)[0])

            probabilities = model.predict_proba(data)[0]
            classes = list(model.classes_)
            fraud_index = classes.index(1)

            fraud_probability = float(
                probabilities[fraud_index]
            )

            transaction_id = transaction.get(
                "transaction_id", "UNKNOWN"
            )

            timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            status = (
                "FRAUD ALERT"
                if prediction == 1
                else "NOT FLAGGED"
            )

            # Update prediction metrics.
            TRANSACTIONS_PREDICTED.labels(
                transaction_type=str(row["type"]),
                prediction=str(prediction),
            ).inc()

            if prediction == 1:
                FRAUD_ALERTS.labels(
                    transaction_type=str(row["type"])
                ).inc()

            # Save every prediction for the dashboard.
            with open(
                LOG_FILE, "a", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file, fieldnames=LOG_COLUMNS
                )
                writer.writerow({
                    "timestamp": timestamp,
                    "transaction_id": transaction_id,
                    "type": row["type"],
                    "amount": row["amount"],
                    "prediction": prediction,
                    "fraud_probability": fraud_probability,
                    "status": status,
                })

            # Store only model-flagged transactions in MongoDB.
            if prediction == 1:
                alert_document = {
                    "transaction_id": transaction_id,
                    "timestamp": timestamp,
                    "type": row["type"],
                    "amount": float(row["amount"]),
                    "step": int(row["step"]),
                    "oldbalanceOrg": float(row["oldbalanceOrg"]),
                    "oldbalanceDest": float(row["oldbalanceDest"]),
                    "amount_to_origin_balance": float(
                        row["amount_to_origin_balance"]
                    ),
                    "prediction": prediction,
                    "fraud_probability": fraud_probability,
                    "status": status,
                }

                # Upsert avoids duplicate alert documents.
                fraud_alerts.update_one(
                    {"transaction_id": transaction_id},
                    {"$set": alert_document},
                    upsert=True,
                )

                print("MongoDB: fraud alert saved!")

            print(f"\n[{timestamp}] {status}")
            print(f"Transaction ID : {transaction_id}")
            print(f"Type           : {row['type']}")
            print(f"Amount         : {row['amount']:.2f}")
            print(f"Fraud score    : {fraud_probability:.2%}")
            print("-" * 60)

        except (KeyError, TypeError, ValueError) as error:
            PROCESSING_ERRORS.labels(
                error_type=type(error).__name__
            ).inc()
            print(f"Invalid transaction skipped: {error}")

        except Exception as error:
            PROCESSING_ERRORS.labels(
                error_type=type(error).__name__
            ).inc()
            print(f"Transaction processing error: {error}")

        finally:
            PROCESSING_TIME.observe(
                time.perf_counter() - processing_start
            )

except KeyboardInterrupt:
    print("\nConsumer stopped by user.")

finally:
    consumer.close()
    mongo_client.close()
    print("Kafka consumer and MongoDB connection closed.")
