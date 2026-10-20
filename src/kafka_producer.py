
import csv
import json
import time
from pathlib import Path
from kafka import KafkaProducer

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed_transactions.csv"

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda data: json.dumps(data).encode("utf-8"),
)

print("\n🚀 REAL-TIME FRAUD PIPELINE")
print("=" * 42)
print("📡 Connected to Kafka: transactions")
print("💳 Streaming sample transactions...\n")

try:
    with DATA_FILE.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for i, row in enumerate(reader):
            if i >= 20:
                break

            transaction = {
                "transaction_id": f"TXN_{i + 1:04d}",
                "step": int(row["step"]),
                "type": row["type"],
                "amount": float(row["amount"]),
                "oldbalanceOrg": float(row["oldbalanceOrg"]),
                "oldbalanceDest": float(row["oldbalanceDest"]),
                "amount_to_origin_balance": float(
                    row["amount_to_origin_balance"]
                ),
                "isFraud": int(row["isFraud"]),
            }

            producer.send("transactions", value=transaction).get(
                timeout=10
            )

            print(
                f"📤 {transaction['transaction_id']} | "
                f"{transaction['type']:<10} | "
                f"Amount: {transaction['amount']:>10,.2f} | "
                f"Label: {transaction['isFraud']}"
            )
            time.sleep(0.5)

    producer.flush()
    print("\n✅ 20 transactions streamed successfully!")

except Exception as error:
    print(f"\n❌ Error: {error}")

finally:
    producer.close()
