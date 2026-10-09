"""
EcoFinOps AI Model Context Protocol (MCP) Server.
Exposes cloud cost, carbon footprint, time-series forecasting, anomaly detection,
and workload scheduling optimization tools for AI agents.

Transport: Stdio
Framework: FastMCP (official mcp Python SDK)
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP

# Ensure current working directory is in sys.path for src imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import generate_hourly_telemetry, generate_synthetic_workloads, REGION_SPECS
from src.dataset_loader import load_public_billing_dataset
from src.carbon_engine import compute_carbon_footprint, get_carbon_methodology_doc
from src.forecasting import prepare_forecasting_data, train_and_evaluate_forecasting, evaluate_rolling_origin_forecasting
from src.anomaly_detector import detect_anomalies_and_waste as core_detect_anomalies
from src.optimizer import optimize_schedules as core_optimize_schedules

# Initialize FastMCP Server
mcp = FastMCP("EcoFinOps AI Server")


def _get_telemetry_dataframe(data_mode: str, num_days: int, seed: int) -> pd.DataFrame:
    """Helper to load telemetry dataframe based on data_mode."""
    if data_mode == "aws_cur":
        return load_public_billing_dataset("aws_cur")
    elif data_mode == "gcp_billing":
        return load_public_billing_dataset("gcp_billing")
    elif data_mode == "synthetic":
        return generate_hourly_telemetry(num_days=num_days, seed=seed)
    else:
        raise ValueError(f"Invalid data_mode: '{data_mode}'. Choose 'synthetic', 'aws_cur', or 'gcp_billing'.")


@mcp.tool()
def get_telemetry_summary(
    num_days: int = 30, seed: int = 42, regions: Optional[List[str]] = None, data_mode: str = "synthetic"
) -> Dict[str, Any]:
    """
    Summarizes cloud fleet spend, energy consumption, and carbon footprint.

    Args:
        num_days: Number of telemetry simulation days (default 30, range 1-60).
        seed: Random seed for telemetry generation reproducibility.
        regions: Optional list of region codes to filter (e.g. ['us-east-1', 'eu-west-1']).
        data_mode: 'synthetic', 'aws_cur', or 'gcp_billing'.

    Returns:
        Dictionary containing total cloud spend ($), carbon footprint (kg CO2e),
        active instance counts, and regional breakdown.
    """
    if num_days < 1 or num_days > 90:
        raise ValueError("num_days must be between 1 and 90 days.")

    df_raw = _get_telemetry_dataframe(data_mode, num_days, seed)
    df = compute_carbon_footprint(df_raw)

    if regions:
        valid_regions = set(REGION_SPECS.keys())
        invalid = [r for r in regions if r not in valid_regions]
        if invalid:
            raise ValueError(f"Invalid region(s): {invalid}. Valid regions are: {list(valid_regions)}")
        df = df[df["region"].isin(regions)].copy()

    total_cost = float(df["hourly_cost_usd"].sum())
    total_carbon = float(df["estimated_kgco2e"].sum())
    total_energy = float(df["energy_consumed_kwh"].sum())
    active_instances = int(df["instance_id"].nunique())

    unmapped_count = int((~df["is_carbon_estimated"]).sum()) if "is_carbon_estimated" in df.columns else 0

    # Regional breakdown
    reg_group = df.groupby("region").agg(
        cost_usd=("hourly_cost_usd", "sum"),
        carbon_kgco2e=("estimated_kgco2e", "sum"),
        active_instances=("instance_id", "nunique"),
    ).reset_index()

    regional_breakdown = {}
    for _, r in reg_group.iterrows():
        regional_breakdown[r["region"]] = {
            "cost_usd": round(float(r["cost_usd"]), 2),
            "carbon_kgco2e": round(float(r["carbon_kgco2e"]), 2),
            "active_instances": int(r["active_instances"]),
        }

    return {
        "data_mode": data_mode,
        "num_days": num_days,
        "seed": seed,
        "telemetry_hours": len(df),
        "total_cost_usd": round(total_cost, 2),
        "total_carbon_kgco2e": round(total_carbon, 2),
        "total_energy_kwh": round(total_energy, 2),
        "active_instances_count": active_instances,
        "unmapped_carbon_records": unmapped_count,
        "regions_included": list(df["region"].unique()),
        "regional_breakdown": regional_breakdown,
    }


@mcp.tool()
def forecast_cloud_spend(
    num_days: int = 30, seed: int = 42, model_type: str = "Ridge", train_ratio: float = 0.8, data_mode: str = "synthetic"
) -> Dict[str, Any]:
    """
    Evaluates out-of-sample time-series cost forecasting comparing an ML model to a 24-hour seasonal naive baseline.

    Args:
        num_days: Simulation window duration in days.
        seed: Random seed for reproducibility.
        model_type: Forecasting algorithm ('Ridge' or 'RandomForest').
        train_ratio: Chronological train/test split ratio (0.5 to 0.9, default 0.8).
        data_mode: 'synthetic', 'aws_cur', or 'gcp_billing'.

    Returns:
        Forecast metrics (MAE, RMSE, MAPE) and test evaluation comparison sample.
    """
    if num_days < 7:
        raise ValueError("num_days must be at least 7 for time-series forecasting lag features.")
    if model_type not in ["Ridge", "RandomForest"]:
        raise ValueError("model_type must be either 'Ridge' or 'RandomForest'.")
    if train_ratio < 0.5 or train_ratio > 0.9:
        raise ValueError("train_ratio must be between 0.5 and 0.9.")

    df_raw = _get_telemetry_dataframe(data_mode, num_days, seed)
    df_carbon = compute_carbon_footprint(df_raw)
    fc_hourly = prepare_forecasting_data(df_carbon)

    if len(fc_hourly) < 10:
        return {
            "data_mode": data_mode,
            "status": "INSUFFICIENT_TIMESTAMPS",
            "message": f"Dataset '{data_mode}' contains insufficient contiguous hourly periods ({len(fc_hourly)} hours after 24h lagging). Time-series ML forecasting requires at least 48 contiguous hourly timestamps.",
            "chronological_80_20_metrics": None,
        }

    eval_df, metrics_summary, _ = train_and_evaluate_forecasting(
        fc_hourly, train_ratio=train_ratio, model_type=model_type
    )

    keys_available = list(metrics_summary.keys())
    ml_model_key = "Ridge Regression"
    if model_type == "RandomForest":
        rf_keys = [k for k in keys_available if "Random Forest" in k]
        if rf_keys:
            ml_model_key = rf_keys[0]

    ml_mae = metrics_summary[ml_model_key]["MAE ($/hr)"]
    base_mae = metrics_summary["Baseline (24h Lag)"]["MAE ($/hr)"]
    improvement_pct = round(((base_mae - ml_mae) / base_mae) * 100, 2)

    rolling_metrics = evaluate_rolling_origin_forecasting(fc_hourly, n_splits=3, test_horizon_hours=48, seed=seed)

    # Sample tail predictions
    sample_preds = []
    for _, row in eval_df.tail(5).iterrows():
        sample_preds.append({
            "timestamp": str(row["timestamp"]),
            "actual_cost_usd": round(float(row["total_hourly_cost"]), 4),
            "baseline_pred_usd": round(float(row["Baseline_Pred"]), 4),
            "ridge_pred_usd": round(float(row["Ridge_Pred"]), 4),
            "random_forest_pred_usd": round(float(row["RandomForest_Pred"]), 4),
        })

    return {
        "data_mode": data_mode,
        "model_choice": model_type,
        "train_ratio": train_ratio,
        "test_samples_count": len(eval_df),
        "chronological_80_20_metrics": metrics_summary,
        "rolling_origin_cv_metrics": rolling_metrics,
        "ml_improvement_over_baseline_pct": improvement_pct,
        "forecast_test_sample": sample_preds,
    }


@mcp.tool()
def detect_anomalies_and_waste(
    num_days: int = 30, seed: int = 42, contamination: float = 0.03, data_mode: str = "synthetic"
) -> Dict[str, Any]:
    """
    Detects high-cost anomaly spikes and identifies zombie / underutilized idle instances.

    Args:
        num_days: Simulation window duration in days.
        seed: Random seed.
        contamination: Anomaly sensitivity fraction (0.01 to 0.10).
        data_mode: 'synthetic', 'aws_cur', or 'gcp_billing'.

    Returns:
        Detailed summary of underutilized instances and top cost anomaly events.
    """
    if num_days < 1:
        raise ValueError("num_days must be at least 1.")
    if contamination < 0.001 or contamination > 0.2:
        raise ValueError("contamination parameter must be between 0.001 and 0.2.")

    df_raw = _get_telemetry_dataframe(data_mode, num_days, seed)
    df_carbon = compute_carbon_footprint(df_raw)
    df_annotated, high_cost_anomalies, idle_summary = core_detect_anomalies(
        df_carbon, contamination=contamination
    )

    idle_list = []
    for _, r in idle_summary.iterrows():
        idle_list.append({
            "instance_id": str(r["instance_id"]),
            "instance_type": str(r["instance_type"]),
            "region": str(r["region"]),
            "idle_hours": float(r["idle_hours"]),
            "wasted_cost_usd": round(float(r["wasted_cost_usd"]), 2),
            "recommendation": str(r["recommendation"]),
        })

    anomalies_list = []
    for _, r in high_cost_anomalies.head(10).iterrows():
        anomalies_list.append({
            "timestamp": str(r["timestamp"]),
            "instance_id": str(r["instance_id"]),
            "instance_type": str(r["instance_type"]),
            "cpu_utilization_pct": round(float(r["cpu_utilization_pct"]), 2),
            "hourly_cost_usd": round(float(r["hourly_cost_usd"]), 4),
            "cost_zscore": round(float(r["cost_zscore"]), 2),
        })

    total_wasted_cost = float(idle_summary["wasted_cost_usd"].sum()) if not idle_summary.empty else 0.0

    return {
        "data_mode": data_mode,
        "zombie_idle_instances_count": len(idle_summary),
        "total_wasted_cost_usd": round(total_wasted_cost, 2),
        "zombie_instances": idle_list,
        "high_cost_anomalies_count": len(high_cost_anomalies),
        "top_cost_anomalies": anomalies_list,
    }


@mcp.tool()
def optimize_workload_schedules(seed: int = 42, data_mode: str = "synthetic") -> Dict[str, Any]:
    """
    Simulates batch workload scheduling optimization comparing Baseline, Cost-Aware, and Carbon-Aware strategies.

    Args:
        seed: Random seed for workload generation.
        data_mode: Mode context tag ('synthetic', 'aws_cur', or 'gcp_billing').

    Returns:
        Strategy performance summary matrix and optimized task start schedules.
    """
    workloads = generate_synthetic_workloads(seed=seed)
    scheduled_tasks, summary_df = core_optimize_schedules(workloads)

    summary_records = summary_df.to_dict(orient="records")

    task_schedules = []
    for _, r in scheduled_tasks.iterrows():
        task_schedules.append({
            "task_id": str(r["task_id"]),
            "name": str(r["name"]),
            "flexibility": str(r["flexibility"]),
            "arrival_hour": int(r["arrival_hour"]),
            "deadline_hour": int(r["deadline_hour"]),
            "baseline_start": int(r["baseline_start"]),
            "cost_aware_start": int(r["cost_aware_start"]),
            "carbon_aware_start": int(r["carbon_aware_start"]),
            "deadline_met": bool(r["deadline_met"]),
        })

    return {
        "data_mode": data_mode,
        "strategy_summary": summary_records,
        "scheduled_tasks_count": len(scheduled_tasks),
        "task_schedules": task_schedules,
    }


@mcp.tool()
def get_carbon_methodology() -> str:
    """
    Returns markdown documentation detailing carbon calculation formulas, PUE assumptions, and grid emission factors.
    """
    return get_carbon_methodology_doc()


if __name__ == "__main__":
    mcp.run()
