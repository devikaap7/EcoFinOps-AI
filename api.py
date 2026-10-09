"""
EcoFinOps AI Python Analytics HTTP API.
Wraps core src/ analytics functions and exposes endpoints for remote MCP gateways.
Enforces ECOFINOPS_API_SECRET Bearer token authentication.
"""

import os
import sys
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import generate_hourly_telemetry, generate_synthetic_workloads, REGION_SPECS
from src.dataset_loader import load_public_billing_dataset
from src.carbon_engine import compute_carbon_footprint, get_carbon_methodology_doc
from src.forecasting import prepare_forecasting_data, train_and_evaluate_forecasting, evaluate_rolling_origin_forecasting
from src.anomaly_detector import detect_anomalies_and_waste as core_detect_anomalies
from src.optimizer import optimize_schedules as core_optimize_schedules

app = FastAPI(title="EcoFinOps AI Analytics API", version="3.0.0")

ECOFINOPS_API_SECRET = os.getenv("ECOFINOPS_API_SECRET", "")


def verify_secret(authorization: Optional[str] = Header(None)):
    """Validates Bearer token authentication against ECOFINOPS_API_SECRET environment variable."""
    if not ECOFINOPS_API_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ECOFINOPS_API_SECRET environment variable is not configured on server."
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. Expected 'Bearer <token>'."
        )
    token = authorization.split("Bearer ", 1)[1].strip()
    if token != ECOFINOPS_API_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Secret token."
        )


def _get_telemetry_df(data_mode: str, num_days: int, seed: int):
    if data_mode == "aws_cur":
        return load_public_billing_dataset("aws_cur")
    elif data_mode == "gcp_billing":
        return load_public_billing_dataset("gcp_billing")
    elif data_mode == "synthetic":
        return generate_hourly_telemetry(num_days=num_days, seed=seed)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid data_mode: '{data_mode}'. Choose 'synthetic', 'aws_cur', or 'gcp_billing'."
        )


# Request Pydantic Schemas
class TelemetrySummaryReq(BaseModel):
    num_days: int = 30
    seed: int = 42
    regions: Optional[List[str]] = None
    data_mode: str = "synthetic"


class ForecastSpendReq(BaseModel):
    num_days: int = 30
    seed: int = 42
    model_type: str = "Ridge"
    train_ratio: float = 0.8
    data_mode: str = "synthetic"


class AnomaliesWasteReq(BaseModel):
    num_days: int = 30
    seed: int = 42
    contamination: float = 0.03
    data_mode: str = "synthetic"


class WorkloadScheduleReq(BaseModel):
    seed: int = 42
    data_mode: str = "synthetic"


# API Endpoints
@app.get("/health")
def health_check():
    """Public health probe for load balancers (No Auth required)."""
    return {"status": "healthy", "service": "EcoFinOps AI Python API"}


@app.post("/api/telemetry_summary")
def get_telemetry_summary_endpoint(req: TelemetrySummaryReq, authorization: Optional[str] = Header(None)):
    verify_secret(authorization)

    if req.num_days < 1 or req.num_days > 90:
        raise HTTPException(status_code=400, detail="num_days must be between 1 and 90 days.")

    df_raw = _get_telemetry_df(req.data_mode, req.num_days, req.seed)
    df = compute_carbon_footprint(df_raw)

    if req.regions:
        valid_regions = set(REGION_SPECS.keys())
        invalid = [r for r in req.regions if r not in valid_regions]
        if invalid:
            raise HTTPException(status_code=400, detail=f"Invalid region(s): {invalid}")
        df = df[df["region"].isin(req.regions)].copy()

    total_cost = float(df["hourly_cost_usd"].sum())
    total_carbon = float(df["estimated_kgco2e"].sum())
    total_energy = float(df["energy_consumed_kwh"].sum())
    active_instances = int(df["instance_id"].nunique())
    unmapped_count = int((~df["is_carbon_estimated"]).sum()) if "is_carbon_estimated" in df.columns else 0

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
        "data_mode": req.data_mode,
        "num_days": req.num_days,
        "seed": req.seed,
        "telemetry_hours": len(df),
        "total_cost_usd": round(total_cost, 2),
        "total_carbon_kgco2e": round(total_carbon, 2),
        "total_energy_kwh": round(total_energy, 2),
        "active_instances_count": active_instances,
        "unmapped_carbon_records": unmapped_count,
        "regions_included": list(df["region"].unique()),
        "regional_breakdown": regional_breakdown,
    }


@app.post("/api/forecast_spend")
def forecast_cloud_spend_endpoint(req: ForecastSpendReq, authorization: Optional[str] = Header(None)):
    verify_secret(authorization)

    if req.num_days < 7:
        raise HTTPException(status_code=400, detail="num_days must be at least 7.")
    if req.model_type not in ["Ridge", "RandomForest"]:
        raise HTTPException(status_code=400, detail="model_type must be 'Ridge' or 'RandomForest'.")
    if req.train_ratio < 0.5 or req.train_ratio > 0.9:
        raise HTTPException(status_code=400, detail="train_ratio must be between 0.5 and 0.9.")

    df_raw = _get_telemetry_df(req.data_mode, req.num_days, req.seed)
    df_carbon = compute_carbon_footprint(df_raw)
    fc_hourly = prepare_forecasting_data(df_carbon)

    if len(fc_hourly) < 10:
        return {
            "data_mode": req.data_mode,
            "status": "INSUFFICIENT_TIMESTAMPS",
            "message": f"Dataset '{req.data_mode}' contains insufficient contiguous hourly periods ({len(fc_hourly)} hours after 24h lagging).",
            "chronological_80_20_metrics": None,
        }

    eval_df, metrics_summary, _ = train_and_evaluate_forecasting(
        fc_hourly, train_ratio=req.train_ratio, model_type=req.model_type
    )

    keys_available = list(metrics_summary.keys())
    ml_model_key = "Ridge Regression"
    if req.model_type == "RandomForest":
        rf_keys = [k for k in keys_available if "Random Forest" in k]
        if rf_keys:
            ml_model_key = rf_keys[0]

    ml_mae = metrics_summary[ml_model_key]["MAE ($/hr)"]
    base_mae = metrics_summary["Baseline (24h Lag)"]["MAE ($/hr)"]
    improvement_pct = round(((base_mae - ml_mae) / base_mae) * 100, 2)

    rolling_metrics = evaluate_rolling_origin_forecasting(fc_hourly, n_splits=3, test_horizon_hours=48, seed=req.seed)

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
        "data_mode": req.data_mode,
        "model_choice": req.model_type,
        "train_ratio": req.train_ratio,
        "test_samples_count": len(eval_df),
        "chronological_80_20_metrics": metrics_summary,
        "rolling_origin_cv_metrics": rolling_metrics,
        "ml_improvement_over_baseline_pct": improvement_pct,
        "forecast_test_sample": sample_preds,
    }


@app.post("/api/anomalies_waste")
def detect_anomalies_and_waste_endpoint(req: AnomaliesWasteReq, authorization: Optional[str] = Header(None)):
    verify_secret(authorization)

    if req.num_days < 1:
        raise HTTPException(status_code=400, detail="num_days must be at least 1.")
    if req.contamination < 0.001 or req.contamination > 0.2:
        raise HTTPException(status_code=400, detail="contamination must be between 0.001 and 0.2.")

    df_raw = _get_telemetry_df(req.data_mode, req.num_days, req.seed)
    df_carbon = compute_carbon_footprint(df_raw)
    df_annotated, high_cost_anomalies, idle_summary = core_detect_anomalies(
        df_carbon, contamination=req.contamination
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
        "data_mode": req.data_mode,
        "zombie_idle_instances_count": len(idle_summary),
        "total_wasted_cost_usd": round(total_wasted_cost, 2),
        "zombie_instances": idle_list,
        "high_cost_anomalies_count": len(high_cost_anomalies),
        "top_cost_anomalies": anomalies_list,
    }


@app.post("/api/optimize_schedules")
def optimize_workload_schedules_endpoint(req: WorkloadScheduleReq, authorization: Optional[str] = Header(None)):
    verify_secret(authorization)

    workloads = generate_synthetic_workloads(seed=req.seed)
    scheduled_tasks, summary_df = core_optimize_schedules(workloads)

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
        "data_mode": req.data_mode,
        "strategy_summary": summary_df.to_dict(orient="records"),
        "scheduled_tasks_count": len(scheduled_tasks),
        "task_schedules": task_schedules,
    }


@app.get("/api/carbon_methodology")
def get_carbon_methodology_endpoint(authorization: Optional[str] = Header(None)):
    verify_secret(authorization)
    return {"methodology_doc": get_carbon_methodology_doc()}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
