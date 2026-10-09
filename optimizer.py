"""
Workload Scheduling Optimizer for EcoFinOps AI.

Compares Baseline (Immediate), Cost-Aware (Min Cost), and Carbon-Aware (Min Carbon)
workload execution schedules over a 24-hour planning window while respecting
arrival times, deadlines, and resource constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def get_hourly_grid_profile() -> pd.DataFrame:
    """
    Generates a 24-hour grid profile showing hourly pricing and carbon intensity multipliers.
    Simulates peak vs off-peak grid dynamics.
    """
    hours = np.arange(24)

    # Carbon intensity peaks at night/evening, drops mid-day (solar peak 10am-4pm)
    carbon_intensity_factor = 1.0 - 0.4 * np.sin((hours - 6) * np.pi / 12)
    carbon_intensity_gco2 = 350.0 * np.clip(carbon_intensity_factor, 0.4, 1.3)

    # Electricity cost peaks during evening (5pm-9pm) and morning (8am-11am)
    cost_factor = 1.0 + 0.5 * np.exp(-((hours - 18) ** 2) / 8.0) + 0.3 * np.exp(-((hours - 9) ** 2) / 6.0)
    base_rate_per_vcpu_hr = 0.04 * cost_factor

    return pd.DataFrame({
        "hour": hours,
        "cost_rate_usd_per_vcpu": np.round(base_rate_per_vcpu_hr, 4),
        "grid_gco2_kwh": np.round(carbon_intensity_gco2, 1),
    })


def optimize_schedules(workloads_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs three scheduling algorithms: Baseline, Cost-Aware, and Carbon-Aware.
    Returns:
    - Task Schedule Breakdown
    - Overall Strategy Comparison Summary
    """
    grid_df = get_hourly_grid_profile()
    grid_dict = grid_df.set_index("hour").to_dict("index")

    scheduled_tasks = []

    for _, row in workloads_df.iterrows():
        task_id = row["task_id"]
        name = row["name"]
        flexibility = row["flexibility"]
        vcpus = row["required_vcpus"]
        ram = row["required_ram_gb"]
        duration = row["duration_hours"]
        arrival = int(row["arrival_hour"])
        deadline = int(row["deadline_hour"])

        # Baseline: Start immediately at arrival
        base_start = arrival
        base_hours = list(range(base_start, min(base_start + duration, 24)))
        base_cost = sum(grid_dict[h]["cost_rate_usd_per_vcpu"] * vcpus for h in base_hours)
        base_carbon = sum(
            (grid_dict[h]["grid_gco2_kwh"] * (vcpus * 25.0 * 1.15 / 1000.0)) for h in base_hours
        ) / 1000.0
        base_met_deadline = (base_start + duration) <= deadline

        # Optimization search window
        if flexibility == "Strict" or (deadline - duration) < arrival:
            # Cannot be shifted
            cost_start = base_start
            carb_start = base_start
        else:
            max_start = min(deadline - duration, 24 - duration)
            valid_starts = list(range(arrival, max_start + 1))

            # Cost-Aware Optimization: Find valid start hour that minimizes total cost
            cost_start = min(
                valid_starts,
                key=lambda s: sum(
                    grid_dict[h]["cost_rate_usd_per_vcpu"] * vcpus for h in range(s, s + duration)
                ),
            )

            # Carbon-Aware Optimization: Find valid start hour that minimizes carbon emissions
            carb_start = min(
                valid_starts,
                key=lambda s: sum(
                    grid_dict[h]["grid_gco2_kwh"] for h in range(s, s + duration)
                ),
            )

        # Cost-Aware evaluation
        cost_hours = list(range(cost_start, cost_start + duration))
        opt_cost = sum(grid_dict[h]["cost_rate_usd_per_vcpu"] * vcpus for h in cost_hours)
        opt_cost_carbon = sum(
            (grid_dict[h]["grid_gco2_kwh"] * (vcpus * 25.0 * 1.15 / 1000.0)) for h in cost_hours
        ) / 1000.0

        # Carbon-Aware evaluation
        carb_hours = list(range(carb_start, carb_start + duration))
        opt_carb_cost = sum(grid_dict[h]["cost_rate_usd_per_vcpu"] * vcpus for h in carb_hours)
        opt_carb_carbon = sum(
            (grid_dict[h]["grid_gco2_kwh"] * (vcpus * 25.0 * 1.15 / 1000.0)) for h in carb_hours
        ) / 1000.0

        # Check deadline compliance for each strategy
        base_met_deadline = (base_start + duration) <= deadline
        cost_met_deadline = (cost_start + duration) <= deadline
        carb_met_deadline = (carb_start + duration) <= deadline

        scheduled_tasks.append({
            "task_id": task_id,
            "name": name,
            "flexibility": flexibility,
            "duration_hours": duration,
            "arrival_hour": arrival,
            "deadline_hour": deadline,
            "baseline_start": base_start,
            "baseline_cost_usd": round(base_cost, 4),
            "baseline_carbon_kgco2e": round(base_carbon, 4),
            "baseline_deadline_met": base_met_deadline,
            "cost_aware_start": cost_start,
            "cost_aware_cost_usd": round(opt_cost, 4),
            "cost_aware_carbon_kgco2e": round(opt_cost_carbon, 4),
            "cost_aware_deadline_met": cost_met_deadline,
            "carbon_aware_start": carb_start,
            "carbon_aware_cost_usd": round(opt_carb_cost, 4),
            "carbon_aware_carbon_kgco2e": round(opt_carb_carbon, 4),
            "carbon_aware_deadline_met": carb_met_deadline,
            "deadline_met": base_met_deadline,
        })

    tasks_df = pd.DataFrame(scheduled_tasks)

    # Strategy Level Comparison with dynamically calculated deadline compliance
    summary = pd.DataFrame([
        {
            "Strategy": "Baseline (Immediate)",
            "Total Cost ($)": round(tasks_df["baseline_cost_usd"].sum(), 2),
            "Total Carbon (kg CO2e)": round(tasks_df["baseline_carbon_kgco2e"].sum(), 4),
            "Deadline Compliance (%)": round(tasks_df["baseline_deadline_met"].mean() * 100, 1),
            "Cost Savings (%)": 0.0,
            "Carbon Reduction (%)": 0.0,
        },
        {
            "Strategy": "Cost-Aware Optimizer",
            "Total Cost ($)": round(tasks_df["cost_aware_cost_usd"].sum(), 2),
            "Total Carbon (kg CO2e)": round(tasks_df["cost_aware_carbon_kgco2e"].sum(), 4),
            "Deadline Compliance (%)": round(tasks_df["cost_aware_deadline_met"].mean() * 100, 1),
            "Cost Savings (%)": round(
                (1 - tasks_df["cost_aware_cost_usd"].sum() / tasks_df["baseline_cost_usd"].sum()) * 100, 1
            ),
            "Carbon Reduction (%)": round(
                (1 - tasks_df["cost_aware_carbon_kgco2e"].sum() / tasks_df["baseline_carbon_kgco2e"].sum()) * 100, 1
            ),
        },
        {
            "Strategy": "Carbon-Aware Optimizer",
            "Total Cost ($)": round(tasks_df["carbon_aware_cost_usd"].sum(), 2),
            "Total Carbon (kg CO2e)": round(tasks_df["carbon_aware_carbon_kgco2e"].sum(), 4),
            "Deadline Compliance (%)": round(tasks_df["carbon_aware_deadline_met"].mean() * 100, 1),
            "Cost Savings (%)": round(
                (1 - tasks_df["carbon_aware_cost_usd"].sum() / tasks_df["baseline_cost_usd"].sum()) * 100, 1
            ),
            "Carbon Reduction (%)": round(
                (1 - tasks_df["carbon_aware_carbon_kgco2e"].sum() / tasks_df["baseline_carbon_kgco2e"].sum()) * 100, 1
            ),
        },
    ])

    return tasks_df, summary
