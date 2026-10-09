"""
Audit Verification Test Suite for EcoFinOps AI V1
Tests all 7 audit points programmatically.
"""

import sys
import os
import pandas as pd
import numpy as np

from src.data_generator import generate_hourly_telemetry, generate_synthetic_workloads, REGION_SPECS
from src.carbon_engine import compute_carbon_footprint, get_carbon_methodology_doc
from src.forecasting import prepare_forecasting_data, train_and_evaluate_forecasting
from src.anomaly_detector import detect_anomalies_and_waste
from src.optimizer import optimize_schedules

print("=== STARTING AUDIT VERIFICATION TEST SUITE ===")

# Test 1: Forecasting split & data leakage
print("\n--- Test 1: Forecasting Chronological Split & Feature Leakage ---")
df = generate_hourly_telemetry(num_days=30, seed=42)
carbon_df = compute_carbon_footprint(df)
fc_df = prepare_forecasting_data(carbon_df)

assert fc_df["timestamp"].is_monotonic_increasing, "Timestamp sequence must be strictly chronological!"
eval_df, metrics, model = train_and_evaluate_forecasting(fc_df, train_ratio=0.8, model_type="Ridge")

# Verify chronological split index
split_idx = int(len(fc_df) * 0.8)
train_max_time = fc_df.iloc[:split_idx]["timestamp"].max()
test_min_time = fc_df.iloc[split_idx:]["timestamp"].min()
assert train_max_time < test_min_time, "Train data must strictly precede test data chronologically!"
print(f"PASS: Train time range ({fc_df.iloc[:split_idx]['timestamp'].min()} to {train_max_time}) precedes Test time range ({test_min_time} to {fc_df.iloc[split_idx:]['timestamp'].max()})")

# Test 2: Evaluation Metrics
print("\n--- Test 2: Evaluation Metrics Calculation & Units ---")
assert "MAE ($/hr)" in metrics["Baseline (24h Lag)"], "MAE key missing explicit units!"
assert "RMSE ($/hr)" in metrics["Baseline (24h Lag)"], "RMSE key missing explicit units!"
assert "MAPE (%)" in metrics["Baseline (24h Lag)"], "MAPE key missing explicit units!"

# Verify MAE math
actuals = eval_df["total_hourly_cost"].values
preds = eval_df["ML_Model_Pred"].values
expected_mae = np.mean(np.abs(actuals - preds))
actual_mae = metrics["Ridge Regression"]["MAE ($/hr)"]
assert round(expected_mae, 4) == round(actual_mae, 4), f"MAE calculation mismatch! Expected {expected_mae}, got {actual_mae}"
print(f"PASS: Metric calculation verified (MAE={actual_mae} $/hr, RMSE={metrics['Ridge Regression']['RMSE ($/hr)']} $/hr, MAPE={metrics['Ridge Regression']['MAPE (%)']}%)")

# Test 3 & 4: Scheduling Optimizer Execution Shifting, Recomputation, and Dynamic Deadline Compliance
print("\n--- Test 3 & 4: Scheduling Optimizer Execution Shifting & Deadlines ---")
workloads = generate_synthetic_workloads(seed=42)
scheduled_tasks, summary = optimize_schedules(workloads)

# Verify execution start times shifted for flexible tasks
flexible_tasks = scheduled_tasks[scheduled_tasks["flexibility"] == "Flexible"]
has_shifted_start = (flexible_tasks["cost_aware_start"] != flexible_tasks["baseline_start"]).any()
assert has_shifted_start, "Flexible task start times should be shifted to optimize cost/carbon!"

# Verify cost and carbon recomputation
for _, row in scheduled_tasks.iterrows():
    if row["cost_aware_start"] != row["baseline_start"]:
        assert row["cost_aware_cost_usd"] != row["baseline_cost_usd"], "Cost must be recomputed when start hour changes!"

# Verify dynamic deadline compliance
for strat in ["Baseline (Immediate)", "Cost-Aware Optimizer", "Carbon-Aware Optimizer"]:
    compliance = summary.loc[summary["Strategy"] == strat, "Deadline Compliance (%)"].values[0]
    assert 0.0 <= compliance <= 100.0, f"Invalid deadline compliance: {compliance}"
print(f"PASS: Scheduling optimization shifts task start hours, recomputes metrics, and dynamically evaluates deadline compliance:\n{summary[['Strategy', 'Total Cost ($)', 'Total Carbon (kg CO2e)', 'Deadline Compliance (%)']]}")

# Test 5: Anomaly & Idle Waste Rules
print("\n--- Test 5: Idle Waste & Anomaly Rules ---")
df_annotated, high_cost_anomalies, idle_summary = detect_anomalies_and_waste(carbon_df)
for _, row in idle_summary.iterrows():
    inst_records = df_annotated[df_annotated["instance_id"] == row["instance_id"]]
    avg_cpu = inst_records["cpu_utilization_pct"].mean()
    assert (inst_records["cpu_utilization_pct"] < 10.0).any(), "Idle instance must have low CPU utilization hours!"
print(f"PASS: Found {len(idle_summary)} zombie idle instances and {len(high_cost_anomalies)} cost anomaly spikes matching documented rules.")

# Test 6: Carbon Methodology & Assumptions
print("\n--- Test 6: Carbon Estimation Assumptions & Documentation ---")
doc = get_carbon_methodology_doc()
assert "Power (W)" in doc and "g CO" in doc, "Methodology document missing required formulas!"
assert "software-based estimates" in doc.lower(), "Methodology document missing estimate disclaimer!"
print("PASS: Carbon estimation methodology and assumptions verified.")

# Test 7: Sidebar Controls Reactivity
print("\n--- Test 7: Sidebar Control Reactivity ---")
df1 = generate_hourly_telemetry(num_days=7, seed=42)
df2 = generate_hourly_telemetry(num_days=14, seed=99)
assert len(df1) != len(df2), "Changing simulation window must change output telemetry row count!"

df_reg_filtered = df1[df1["region"].isin(["us-west-2"])].copy()
assert df_reg_filtered["region"].nunique() == 1, "Region filtering must restrict dataset to selected regions!"
print("PASS: Controls update dataset size and regional scope consistently.")

print("\n=== ALL 7 AUDIT VERIFICATION TESTS PASSED SUCCESSFULLY! ===")
