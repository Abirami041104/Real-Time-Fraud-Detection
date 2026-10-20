# &#x20;               FRUAD DETECTION TOOL

&#x20;                                               ( **program running commands)**



Terminal 1 — Start Kafka

cd D:\\kafka

.\\bin\\windows\\kafka-server-start.bat .\\config\\server.properties





Keep this terminal running.









Terminal 2 — Start the fraud detection consumer

cd D:\\Real-Time-Fraud-Detection

.\\venv\\Scripts\\Activate.ps1

python .\\src\\kafka\_consumer.py





Loads the model, consumes transactions, and exposes Prometheus metrics on port 8000.









Terminal 3 — Start Prometheus

cd D:\\prometheus\\prometheus-3.15.0.windows-amd64

.\\prometheus.exe --config.file=prometheus.yml





Collects the metrics exposed by the consumer.









Terminal 4 — Start Streamlit

cd D:\\Real-Time-Fraud-Detection

.\\venv\\Scripts\\Activate.ps1

python -m streamlit run dashboard.py





Opens the main application dashboard at port 8501.







Grafana is installed separately and can usually be opened at http://localhost:3000 through its Windows service. Start it through Windows Services if it isn't running.



Once the services are ready, run the producer in another terminal to send test transactions:

cd D:\\Real-Time-Fraud-Detection

.\\venv\\Scripts\\Activate.ps1

python .\\src\\kafka\_producer.py

