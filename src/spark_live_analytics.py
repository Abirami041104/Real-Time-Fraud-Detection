import json
import time

from kafka import KafkaConsumer
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, sum as spark_sum
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType
)

BATCH_SIZE = 10
BATCH_TIMEOUT_SECONDS = 10

schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("type", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("oldbalanceOrg", DoubleType(), True),
    StructField("oldbalanceDest", DoubleType(), True),
    StructField("amount_to_origin_balance", DoubleType(), True),
])

def main():
    spark = (
        SparkSession.builder
        .appName("FraudDetectionLiveAnalytics")
        .master("local[2]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    consumer = None

    try:
        consumer = KafkaConsumer(
            "transactions",
            bootstrap_servers=["localhost:9092"],
            value_deserializer=lambda message: json.loads(
                message.decode("utf-8")
            ),
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            group_id="spark-analytics-test-v2",
        )

        print("=" * 55)
        print("SPARK LIVE TRANSACTION ANALYTICS")
        print("Connected to Kafka topic: transactions")
        print(f"Processing batches of {BATCH_SIZE} transactions")
        print("Start the producer in another terminal.")
        print("Press Ctrl+C to stop.")
        print("=" * 55)

        batch = []
        last_batch_time = time.time()

        while True:
            records = consumer.poll(timeout_ms=1000, max_records=100)

            for messages in records.values():
                for message in messages:
                    transaction = message.value

                    try:
                        batch.append({
                            "transaction_id": str(
                                transaction.get("transaction_id", "UNKNOWN")
                            ),
                            "type": str(transaction["type"]),
                            "amount": float(transaction["amount"]),
                            "oldbalanceOrg": float(
                                transaction.get("oldbalanceOrg", 0)
                            ),
                            "oldbalanceDest": float(
                                transaction.get("oldbalanceDest", 0)
                            ),
                            "amount_to_origin_balance": float(
                                transaction.get(
                                    "amount_to_origin_balance", 0
                                )
                            ),
                        })
                    except (KeyError, TypeError, ValueError) as error:
                        print(f"Skipping invalid transaction: {error}")

            timed_out = (
                time.time() - last_batch_time >= BATCH_TIMEOUT_SECONDS
            )

            if batch and (len(batch) >= BATCH_SIZE or timed_out):
                df = spark.createDataFrame(batch, schema=schema)

                print(f"\nProcessed batch: {len(batch)} transactions")
                print("Transaction-type summary:")

                (
                    df.groupBy("type")
                    .agg(
                        count("*").alias("transaction_count"),
                        spark_sum("amount").alias("total_amount"),
                        avg("amount_to_origin_balance").alias(
                            "avg_amount_balance_ratio"
                        ),
                    )
                    .orderBy("type")
                    .show(truncate=False)
                )

                batch.clear()
                last_batch_time = time.time()

    except KeyboardInterrupt:
        print("\nSpark live analytics stopped.")

    finally:
        if consumer is not None:
            consumer.close()
        spark.stop()
        print("Spark and Kafka connections closed.")

if __name__ == "__main__":
    main()
