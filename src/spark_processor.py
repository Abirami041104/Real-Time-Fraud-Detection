from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as spark_sum

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed_transactions.csv"

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_FILE}")

    spark = (
        SparkSession.builder
        .appName("FinancialFraudDetection")
        .master("local[2]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        print("\nReading transaction sample with Spark...")

        transactions = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(str(DATA_FILE))
            .limit(10000)
        )

        print("\nSample schema:")
        transactions.printSchema()

        print("\nTransaction types and counts:")
        transactions.groupBy("type").agg(
            count("*").alias("transaction_count"),
            spark_sum("amount").alias("total_amount")
        ).orderBy(col("transaction_count").desc()).show(truncate=False)

        print("\nFraud summary in this sample:")
        transactions.groupBy("isFraud").count().show()

        print("\nSpark transaction processing completed successfully!")

    finally:
        spark.stop()

if __name__ == "__main__":
    main()
