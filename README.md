# Real-Time Financial Fraud Detection Pipeline

## Project Overview

The Real-Time Financial Fraud Detection Pipeline is a machine learning project that identifies potentially fraudulent financial transactions as they are streamed through Apache Kafka.

A trained Random Forest model predicts whether a transaction should be flagged. The system stores prediction history in CSV, displays transaction results through Streamlit, and exposes monitoring metrics to Prometheus for visualization in Grafana.

This project demonstrates real-time message streaming, machine learning inference, transaction monitoring, and fraud-alert tracking.

## Key Features

- Transaction preprocessing and feature engineering
- Fraud classification using a trained Random Forest model
- Real-time transaction streaming with Apache Kafka
- Fraud prediction and alert generation
- Prediction history stored in CSV
- Streamlit dashboard for transaction monitoring
- Interactive transaction testing
- Prometheus metrics collection and Grafana visualization
- Optional MongoDB storage for fraud alerts, if configured and running

## Technologies Used

- **Language:** Python
- **Machine Learning:** Scikit-learn, Pandas, NumPy
- **Streaming:** Apache Kafka
- **Dashboards:** Streamlit, Plotly, Grafana
- **Monitoring:** Prometheus, Prometheus Client
- **Model Persistence:** Joblib
- **Optional Database:** MongoDB

## System Architecture

1. **Kafka Producer:** Publishes sample financial transactions to the `transactions` topic.
2. **Kafka Consumer:** Reads transactions from Kafka and prepares the required model features.
3. **Machine Learning Model:** Predicts whether each transaction should be flagged.
4. **Prediction Storage:** Records prediction results in `data/live_predictions.csv`.
5. **Streamlit Dashboard:** Displays transaction results, fraud predictions, and visual summaries.
6. **Prometheus:** Collects transaction-processing metrics exposed by the consumer.
7. **Grafana:** Visualizes transaction counts, fraud alerts, transaction types, and processing errors.

## Model Inputs

The trained model expects the following features:

- `step`
- `type`
- `amount`
- `oldbalanceOrg`
- `oldbalanceDest`
- `amount_to_origin_balance`

The engineered feature is calculated as:

`amount / (oldbalanceOrg + 1)`

The model produces a prediction:

- `0` — Not flagged by the model
- `1` — Flagged as potentially fraudulent

A prediction is a model assessment, not proof that a transaction is fraudulent or safe.

## Project Structure

```text
Real-Time-Fraud-Detection/
├── dashboards/
├── data/
├── logs/
├── models/
│   └── fraud_detection_model.pkl
├── reports/
├── src/
│   ├── kafka_producer.py
│   ├── kafka_consumer.py
│   └── spark_live_analytics.py
├── dashboard.py
├── fraud_tester.py
├── requirements.txt
├── README.md
└── .gitignore
```

Some additional scripts and configuration files may exist in the local development environment.

## Prerequisites

- Python 3.11 or a compatible version
- Java, as required by your Kafka distribution
- Apache Kafka
- Prometheus
- Grafana
- The trained model file in `models/`
- Required Python packages from `requirements.txt`

## Setup

Open PowerShell and navigate to the project directory:

```powershell
cd D:\Real-Time-Fraud-Detection
```

Create and activate a virtual environment if you have not already created one:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Use the existing trained model at `models/fraud_detection_model.pkl`. Model training and data preparation are separate steps and are not required every time the pipeline runs.

## Running the Project on Windows

Start each long-running service in a separate terminal.

### 1. Start Kafka

```powershell
cd D:\kafka
.\bin\windows\kafka-server-start.bat .\config\server.properties
```

Ensure that the Kafka broker is ready and the `transactions` topic exists.

### 2. Start the Kafka Consumer

```powershell
cd D:\Real-Time-Fraud-Detection
.\venv\Scripts\Activate.ps1
python .\src\kafka_consumer.py
```

Keep this terminal running. The consumer reads messages, predicts fraud, saves prediction history, and exposes monitoring metrics.

### 3. Start Prometheus

```powershell
cd D:\prometheus\prometheus-3.15.0.windows-amd64
.\prometheus.exe --config.file=prometheus.yml
```

Confirm that the configured fraud-detection target is available at `localhost:8000`.

### 4. Start the Streamlit Dashboard

```powershell
cd D:\Real-Time-Fraud-Detection
.\venv\Scripts\Activate.ps1
python -m streamlit run dashboard.py
```

Open: http://localhost:8501

### 5. Start the Transaction Producer

In another terminal:

```powershell
cd D:\Real-Time-Fraud-Detection
.\venv\Scripts\Activate.ps1
python .\src\kafka_producer.py
```

The producer publishes sample transactions for the consumer to process.

### 6. Open Grafana

Ensure the Grafana Windows service is running and that its Prometheus data source is configured.

Open: http://localhost:3000

Select the **Real Time Fraud Detection Tool** dashboard.

### 7. Open the Interactive Tester

To run the separate transaction tester:

```powershell
cd D:\Real-Time-Fraud-Detection
.\venv\Scripts\Activate.ps1
python -m streamlit run fraud_tester.py --server.port 8502
```

Open: http://localhost:8502

The tester is a separate interface; do not assume its submissions enter Kafka unless that integration is implemented.

## Monitoring Metrics

The Kafka consumer exposes metrics on port `8000`, which Prometheus scrapes.

Examples include:

- Total processed transaction predictions
- Fraud-alert counts
- Processing errors
- Transaction-processing duration

Prometheus interface: http://localhost:9090

Metrics endpoint: http://localhost:8000/metrics

## Important Limitations

- The project uses the synthetic PaySim financial transaction dataset.
- Synthetic-data patterns may not represent fraud in real financial institutions.
- High validation scores on this dataset do not guarantee comparable performance on real-world transactions.
- A transaction marked `NOT FLAGGED` may still be fraudulent.
- The fraud probability is a model estimate and should not be interpreted as a guaranteed real-world risk.
- The pipeline is an educational demonstration, not a production-ready fraud prevention service.

## Future Improvements

- Add automated tests and stronger error handling.
- Improve alert investigation and historical reporting.
- Evaluate model performance on independent, realistic datasets.
- Add secure configuration and deployment practices.
- Expand database integration and operational monitoring.

## Disclaimer

This project is intended for educational and demonstration purposes only. It should not be used as the sole basis for real financial decisions or production fraud prevention.