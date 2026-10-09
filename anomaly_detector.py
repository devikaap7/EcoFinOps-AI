"""
Anomaly Detection & Idle Waste Identifier for EcoFinOps AI.

Identifies:
1. Unusually high hourly cost spikes using Isolation Forest & Z-Score analysis.
2. Underutilized / Zombie resources (low CPU/RAM usage while accumulating cost).
"""

import numpy as np
import pandas as pd
from typing import Tuple


def detect_anomalies_and_waste(
    df: pd.DataFrame, contamination: float = 0.03
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyzes telemetry data for cost anomalies and underutilized instances.
    Returns:
    - Annotated DataFrame with anomaly flags
    - DataFrame of High Cost Anomalies
    - DataFrame of Underutilized / Idle Instances
    """
    df = df.copy()

    # 1. Isolation Forest for Multivariate Anomaly Detection (CPU, Cost, Memory)
    try:
        from sklearn.ensemble import IsolationForest
        feature_matrix = df[["cpu_utilization_pct", "memory_utilization_pct", "hourly_cost_usd"]].values
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        iso_scores = iso_forest.fit_predict(feature_matrix)
        df["is_iso_anomaly"] = (iso_scores == -1)
    except (ImportError, Exception):
        # Fallback to statistical Z-score anomaly detection if Windows Application Control blocks compiled tree DLL (_tree.pyd)
        cost_z = (df["hourly_cost_usd"] - df["hourly_cost_usd"].mean()) / (df["hourly_cost_usd"].std() + 1e-6)
        cpu_z = (df["cpu_utilization_pct"] - df["cpu_utilization_pct"].mean()) / (df["cpu_utilization_pct"].std() + 1e-6)
        df["is_iso_anomaly"] = (cost_z.abs() > 2.5) | (cpu_z.abs() > 2.5)

    # 2. Rule-Based Thresholds for High Cost Spikes
    mean_cost = df["hourly_cost_usd"].mean()
    std_cost = df["hourly_cost_usd"].std()
    df["cost_zscore"] = (df["hourly_cost_usd"] - mean_cost) / (std_cost + 1e-6)
    df["is_cost_spike"] = (df["cost_zscore"] > 2.5) & (df["cpu_utilization_pct"] > 85.0)

    # Combined Cost Anomaly Flag
    df["is_anomaly"] = df["is_iso_anomaly"] | df["is_cost_spike"]

    # 3. Rule-Based Waste Detection: Zombie / Underutilized Resources
    # Idle defined as CPU < 10% for extended hours while incurring > $0.03/hr cost
    df["is_underutilized"] = (df["cpu_utilization_pct"] < 10.0) & (df["hourly_cost_usd"] > 0.03)

    # Filter High Cost Anomalies
    high_cost_anomalies = (
        df[df["is_anomaly"]]
        .sort_values("hourly_cost_usd", ascending=False)
        .reset_index(drop=True)
    )

    # Filter Underutilized Resources aggregated by instance
    idle_summary = (
        df[df["is_underutilized"]]
        .groupby(["instance_id", "instance_type", "region"])
        .agg(
            idle_hours=("usage_hours", "sum"),
            wasted_cost_usd=("hourly_cost_usd", "sum"),
            wasted_kgco2e=("estimated_kgco2e", "sum"),
            avg_cpu_pct=("cpu_utilization_pct", "mean"),
        )
        .reset_index()
        .sort_values("wasted_cost_usd", ascending=False)
    )

    idle_summary["recommendation"] = idle_summary.apply(
        lambda r: f"Terminate or Downsize {r['instance_id']} ({r['instance_type']}): {r['idle_hours']} idle hours, ${r['wasted_cost_usd']:.2f} wasted.",
        axis=1,
    )

    return df, high_cost_anomalies, idle_summary
