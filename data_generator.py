"""
Synthetic Cloud Workload & Billing Data Generator for EcoFinOps AI.
Generates realistic hourly telemetry for cost forecasting, anomaly detection,
carbon accounting, and workload scheduling.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

# Configuration dictionaries for instance types and regions
INSTANCE_SPECS: Dict[str, Dict[str, float]] = {
    "t3.medium": {
        "vcpus": 2,
        "ram_gb": 4.0,
        "base_cost_per_hr": 0.0416,
        "tdp_watts": 25.0,
        "idle_watts": 8.0,
    },
    "c5.xlarge": {
        "vcpus": 4,
        "ram_gb": 8.0,
        "base_cost_per_hr": 0.1700,
        "tdp_watts": 110.0,
        "idle_watts": 35.0,
    },
    "r5.2xlarge": {
        "vcpus": 8,
        "ram_gb": 64.0,
        "base_cost_per_hr": 0.5040,
        "tdp_watts": 210.0,
        "idle_watts": 60.0,
    },
    "g4dn.xlarge": {
        "vcpus": 4,
        "ram_gb": 16.0,
        "base_cost_per_hr": 0.5260,
        "tdp_watts": 300.0,
        "idle_watts": 90.0,
    },
}

REGION_SPECS: Dict[str, Dict[str, float]] = {
    "us-east-1": {"pue": 1.15, "grid_gco2_kwh": 385.0, "name": "US East (N. Virginia)"},
    "us-west-2": {"pue": 1.12, "grid_gco2_kwh": 120.0, "name": "US West (Oregon)"},
    "eu-west-1": {"pue": 1.18, "grid_gco2_kwh": 260.0, "name": "Europe (Ireland)"},
    "ap-south-1": {"pue": 1.22, "grid_gco2_kwh": 708.0, "name": "AP (Mumbai)"},
}


def generate_hourly_telemetry(
    num_days: int = 30, seed: int = 42
) -> pd.DataFrame:
    """
    Generates realistic hourly telemetry data across multiple instances and regions.
    """
    np.random.seed(seed)

    start_date = pd.Timestamp("2026-09-01 00:00:00")
    total_hours = num_days * 24
    time_index = pd.date_range(start=start_date, periods=total_hours, freq="h")

    # Define fleet of instances
    fleet = [
        {"instance_id": "inst-web-01", "instance_type": "t3.medium", "region": "us-east-1", "pattern": "web_server"},
        {"instance_id": "inst-web-02", "instance_type": "t3.medium", "region": "us-west-2", "pattern": "web_server"},
        {"instance_id": "inst-app-01", "instance_type": "c5.xlarge", "region": "us-east-1", "pattern": "app_compute"},
        {"instance_id": "inst-app-02", "instance_type": "c5.xlarge", "region": "eu-west-1", "pattern": "app_compute"},
        {"instance_id": "inst-db-01", "instance_type": "r5.2xlarge", "region": "us-east-1", "pattern": "database"},
        {"instance_id": "inst-gpu-01", "instance_type": "g4dn.xlarge", "region": "ap-south-1", "pattern": "batch_ml"},
        {"instance_id": "inst-idle-zombie", "instance_type": "r5.2xlarge", "region": "us-west-2", "pattern": "zombie_idle"},
        {"instance_id": "inst-burst-etl", "instance_type": "c5.xlarge", "region": "ap-south-1", "pattern": "spiky_etl"},
    ]

    records = []

    for inst in fleet:
        inst_id = inst["instance_id"]
        inst_type = inst["instance_type"]
        region = inst["region"]
        pattern = inst["pattern"]
        specs = INSTANCE_SPECS[inst_type]
        reg_specs = REGION_SPECS[region]

        for i, dt in enumerate(time_index):
            hour = dt.hour
            day_of_week = dt.dayofweek

            # Base diurnal curve (peak 10am - 4pm)
            diurnal = np.sin((hour - 6) * np.pi / 12)  # -1 to +1
            diurnal_norm = (diurnal + 1) / 2  # 0 to 1

            # Business day factor (Mon=0..Fri=4)
            is_weekend = 1 if day_of_week >= 5 else 0
            weekday_mult = 0.4 if is_weekend else 1.0

            if pattern == "web_server":
                cpu = 20 + 40 * diurnal_norm * weekday_mult + np.random.normal(0, 5)
                mem = 40 + 20 * diurnal_norm + np.random.normal(0, 3)
            elif pattern == "app_compute":
                cpu = 30 + 50 * diurnal_norm * weekday_mult + np.random.normal(0, 8)
                mem = 50 + 25 * diurnal_norm + np.random.normal(0, 4)
            elif pattern == "database":
                cpu = 45 + 30 * diurnal_norm + np.random.normal(0, 4)
                mem = 75 + 10 * diurnal_norm + np.random.normal(0, 2)
            elif pattern == "batch_ml":
                # Intermittent heavy ML bursts
                is_active = (hour in [2, 3, 4, 14, 15, 16])
                cpu = (85 + np.random.normal(0, 5)) if is_active else (5 + np.random.normal(0, 2))
                mem = (80 + np.random.normal(0, 4)) if is_active else (15 + np.random.normal(0, 2))
            elif pattern == "zombie_idle":
                # Forgotten instance running at 3-7% CPU
                cpu = 4 + np.random.normal(0, 1.5)
                mem = 60 + np.random.normal(0, 1.0)
            elif pattern == "spiky_etl":
                # High cost anomaly spikes randomly
                is_spike = np.random.random() < 0.03
                cpu = (98 + np.random.normal(0, 1)) if is_spike else (15 + np.random.normal(0, 3))
                mem = (90 + np.random.normal(0, 2)) if is_spike else (30 + np.random.normal(0, 3))
            else:
                cpu = 30 + np.random.normal(0, 5)
                mem = 40 + np.random.normal(0, 5)

            cpu = float(np.clip(cpu, 2.0, 99.5))
            mem = float(np.clip(mem, 5.0, 98.0))

            # Hourly cost calculation (base rate + data transfer/iops scaling)
            # Add anomaly multiplier for spiky_etl spikes
            cost_multiplier = 2.5 if (pattern == "spiky_etl" and cpu > 90) else 1.0
            hourly_cost = round(specs["base_cost_per_hr"] * (0.8 + 0.4 * (cpu / 100)) * cost_multiplier, 4)

            records.append({
                "timestamp": dt,
                "instance_id": inst_id,
                "instance_type": inst_type,
                "region": region,
                "pattern": pattern,
                "cpu_utilization_pct": round(cpu, 2),
                "memory_utilization_pct": round(mem, 2),
                "usage_hours": 1.0,
                "hourly_cost_usd": hourly_cost,
                "vcpus": specs["vcpus"],
                "ram_gb": specs["ram_gb"],
                "tdp_watts": specs["tdp_watts"],
                "idle_watts": specs["idle_watts"],
                "pue": reg_specs["pue"],
                "grid_gco2_kwh": reg_specs["grid_gco2_kwh"],
            })

    return pd.DataFrame(records)


def generate_synthetic_workloads(seed: int = 42) -> pd.DataFrame:
    """
    Generates a list of batch/flexible workloads to be optimized by the scheduler.
    """
    np.random.seed(seed)

    workloads = [
        {"task_id": "task-etl-daily", "name": "Daily Analytics ETL", "workload_type": "Batch", "flexibility": "Flexible", "required_vcpus": 4, "required_ram_gb": 16, "duration_hours": 3, "arrival_hour": 1, "deadline_hour": 10},
        {"task_id": "task-ml-train", "name": "Model Retraining Pipeline", "workload_type": "Batch", "flexibility": "Flexible", "required_vcpus": 8, "required_ram_gb": 32, "duration_hours": 4, "arrival_hour": 2, "deadline_hour": 16},
        {"task_id": "task-backup", "name": "Database Backup & Sync", "workload_type": "Batch", "flexibility": "Flexible", "required_vcpus": 2, "required_ram_gb": 8, "duration_hours": 2, "arrival_hour": 0, "deadline_hour": 8},
        {"task_id": "task-report", "name": "Financial Report Gen", "workload_type": "Batch", "flexibility": "Flexible", "required_vcpus": 4, "required_ram_gb": 16, "duration_hours": 2, "arrival_hour": 6, "deadline_hour": 14},
        {"task_id": "task-web-core", "name": "Core Web Service", "workload_type": "Interactive", "flexibility": "Strict", "required_vcpus": 4, "required_ram_gb": 8, "duration_hours": 24, "arrival_hour": 0, "deadline_hour": 24},
        {"task_id": "task-log-indexer", "name": "Log Compression Job", "workload_type": "Background", "flexibility": "Flexible", "required_vcpus": 2, "required_ram_gb": 4, "duration_hours": 3, "arrival_hour": 3, "deadline_hour": 18},
    ]

    return pd.DataFrame(workloads)
