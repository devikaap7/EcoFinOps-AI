import sys
import os

# Test importing all modules and executing pipeline
from src.data_generator import generate_hourly_telemetry, generate_synthetic_workloads
from src.carbon_engine import compute_carbon_footprint
from src.forecasting import prepare_forecasting_data, train_and_evaluate_forecasting
from src.anomaly_detector import detect_anomalies_and_waste
from src.optimizer import optimize_schedules

print("Testing Data Generator...")
telemetry_df = generate_hourly_telemetry(num_days=7, seed=42)
print(f"Generated {len(telemetry_df)} telemetry rows.")

print("Testing Carbon Engine...")
carbon_df = compute_carbon_footprint(telemetry_df)
print(f"Computed carbon footprint. Total kgCO2e: {carbon_df['estimated_kgco2e'].sum():.2f}")

print("Testing Forecasting Pipeline...")
fc_data = prepare_forecasting_data(carbon_df)
eval_df, metrics, model = train_and_evaluate_forecasting(fc_data, model_type="RandomForest")
print("Forecasting Metrics:", metrics)

print("Testing Anomaly Detector...")
df, anomalies, idle = detect_anomalies_and_waste(carbon_df)
print(f"Found {len(anomalies)} cost anomalies and {len(idle)} idle resources.")

print("Testing Optimizer...")
workloads_df = generate_synthetic_workloads(seed=42)
tasks_df, summary = optimize_schedules(workloads_df)
print("Optimizer Summary:")
print(summary)

print("ALL Core ML & Engine Pipelines PASS! [SUCCESS]")
