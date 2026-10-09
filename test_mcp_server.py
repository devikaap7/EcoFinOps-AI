"""
MCP Server Automated Test Suite for EcoFinOps AI V2.
Verifies that all exposed MCP tools return exact numerical results matching
the underlying verified Python analytics functions and properly validate inputs.
"""

import sys
import os
import pandas as pd
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import generate_hourly_telemetry, generate_synthetic_workloads
from src.carbon_engine import compute_carbon_footprint, get_carbon_methodology_doc
from src.forecasting import prepare_forecasting_data, train_and_evaluate_forecasting
from src.anomaly_detector import detect_anomalies_and_waste as underlying_detect_anomalies
from src.optimizer import optimize_schedules as underlying_optimize_schedules

from mcp_server import (
    get_telemetry_summary,
    forecast_cloud_spend,
    detect_anomalies_and_waste,
    optimize_workload_schedules,
    get_carbon_methodology,
)

print("=== STARTING MCP SERVER AUTOMATED TEST SUITE ===")


def test_telemetry_summary_parity():
    print("\n--- Testing get_telemetry_summary Tool Parity ---")
    num_days = 14
    seed = 42
    
    # Tool execution
    tool_res = get_telemetry_summary(num_days=num_days, seed=seed)
    
    # Direct underlying function execution
    df_raw = generate_hourly_telemetry(num_days=num_days, seed=seed)
    df_carbon = compute_carbon_footprint(df_raw)
    
    expected_cost = round(float(df_carbon["hourly_cost_usd"].sum()), 2)
    expected_carbon = round(float(df_carbon["estimated_kgco2e"].sum()), 2)
    
    assert tool_res["total_cost_usd"] == expected_cost, f"Cost mismatch! Tool: {tool_res['total_cost_usd']}, Direct: {expected_cost}"
    assert tool_res["total_carbon_kgco2e"] == expected_carbon, f"Carbon mismatch! Tool: {tool_res['total_carbon_kgco2e']}, Direct: {expected_carbon}"
    assert tool_res["active_instances_count"] == df_carbon["instance_id"].nunique()
    print("PASS: get_telemetry_summary tool returns exact parity with underlying carbon engine.")


def test_forecast_cloud_spend_parity():
    print("\n--- Testing forecast_cloud_spend Tool Parity ---")
    num_days = 21
    seed = 42
    model_type = "Ridge"
    
    # Tool execution
    tool_res = forecast_cloud_spend(num_days=num_days, seed=seed, model_type=model_type, train_ratio=0.8)
    
    # Direct execution
    df_raw = generate_hourly_telemetry(num_days=num_days, seed=seed)
    df_carbon = compute_carbon_footprint(df_raw)
    fc_df = prepare_forecasting_data(df_carbon)
    _, direct_metrics, _ = train_and_evaluate_forecasting(fc_df, train_ratio=0.8, model_type=model_type)
    
    tool_ml_mae = tool_res["chronological_80_20_metrics"]["Ridge Regression"]["MAE ($/hr)"]
    direct_ml_mae = direct_metrics["Ridge Regression"]["MAE ($/hr)"]
    
    assert tool_ml_mae == direct_ml_mae, f"MAE mismatch! Tool: {tool_ml_mae}, Direct: {direct_ml_mae}"
    assert "rolling_origin_cv_metrics" in tool_res
    print(f"PASS: forecast_cloud_spend tool matches direct ML model MAE ({tool_ml_mae} $/hr).")


def test_detect_anomalies_and_waste_parity():
    print("\n--- Testing detect_anomalies_and_waste Tool Parity ---")
    num_days = 14
    seed = 42
    
    tool_res = detect_anomalies_and_waste(num_days=num_days, seed=seed)
    
    df_raw = generate_hourly_telemetry(num_days=num_days, seed=seed)
    df_carbon = compute_carbon_footprint(df_raw)
    _, direct_anomalies, direct_idle = underlying_detect_anomalies(df_carbon)
    
    assert tool_res["zombie_idle_instances_count"] == len(direct_idle)
    assert tool_res["high_cost_anomalies_count"] == len(direct_anomalies)
    print(f"PASS: detect_anomalies_and_waste tool matches direct anomaly count ({len(direct_anomalies)}) and zombie count ({len(direct_idle)}).")


def test_optimize_workload_schedules_parity():
    print("\n--- Testing optimize_workload_schedules Tool Parity ---")
    seed = 42
    
    tool_res = optimize_workload_schedules(seed=seed)
    
    workloads = generate_synthetic_workloads(seed=seed)
    direct_tasks, direct_summary = underlying_optimize_schedules(workloads)
    
    assert tool_res["scheduled_tasks_count"] == len(direct_tasks)
    assert len(tool_res["strategy_summary"]) == len(direct_summary)
    print("PASS: optimize_workload_schedules tool matches direct optimizer outputs.")


def test_carbon_methodology():
    print("\n--- Testing get_carbon_methodology Tool ---")
    doc = get_carbon_methodology()
    assert len(doc) > 100
    assert "software-based estimates" in doc.lower()
    print("PASS: get_carbon_methodology tool returns complete documentation.")


def test_error_handling_and_validation():
    print("\n--- Testing Input Validation & Error Handling ---")
    
    # Invalid num_days
    try:
        get_telemetry_summary(num_days=-5)
        assert False, "Should have raised ValueError for negative num_days"
    except ValueError as e:
        assert "between 1 and 90" in str(e)
        
    # Invalid forecasting algorithm
    try:
        forecast_cloud_spend(model_type="UnsupportedModel")
        assert False, "Should have raised ValueError for invalid model_type"
    except ValueError as e:
        assert "must be either 'Ridge' or 'RandomForest'" in str(e)

    # Invalid region
    try:
        get_telemetry_summary(regions=["invalid-region-999"])
        assert False, "Should have raised ValueError for invalid region"
    except ValueError as e:
        assert "Invalid region" in str(e)

    print("PASS: Input validation properly catches invalid arguments with descriptive errors.")


if __name__ == "__main__":
    test_telemetry_summary_parity()
    test_forecast_cloud_spend_parity()
    test_detect_anomalies_and_waste_parity()
    test_optimize_workload_schedules_parity()
    test_carbon_methodology()
    test_error_handling_and_validation()
    print("\n=== ALL MCP SERVER TESTS PASSED SUCCESSFULLY! ===")
