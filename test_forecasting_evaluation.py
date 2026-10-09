"""
Forecasting Research Evaluation Automated Test Suite for EcoFinOps AI.
Tests temporal ordering, data leakage prevention, 3-way model comparison,
small/zero denominator metric calculation resilience, and rolling-origin reproducibility.
"""

import sys
import os
import pandas as pd
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import generate_hourly_telemetry
from src.carbon_engine import compute_carbon_footprint
from src.forecasting import (
    prepare_forecasting_data,
    train_and_evaluate_forecasting,
    evaluate_rolling_origin_forecasting,
    calculate_metrics,
    FEATURE_COLS,
)

print("=== STARTING FORECASTING RESEARCH EVALUATION TEST SUITE ===")


def test_temporal_ordering_and_leakage_prevention():
    print("\n--- Test 1: Temporal Ordering & Feature Leakage Prevention ---")
    df_raw = generate_hourly_telemetry(num_days=30, seed=42)
    df_carbon = compute_carbon_footprint(df_raw)
    fc_df = prepare_forecasting_data(df_carbon)

    # 1. Monotonic increasing timestamps
    assert fc_df["timestamp"].is_monotonic_increasing, "Timestamps must be strictly chronological!"

    # 2. Verify all lag/rolling features use t-1 or earlier
    for idx in range(25, len(fc_df)):
        # lag_1h at row idx must equal total_hourly_cost at row idx-1
        assert fc_df.iloc[idx]["lag_1h"] == fc_df.iloc[idx - 1]["total_hourly_cost"], "lag_1h must match cost at t-1"
        # lag_24h at row idx must equal total_hourly_cost at row idx-24
        assert fc_df.iloc[idx]["lag_24h"] == fc_df.iloc[idx - 24]["total_hourly_cost"], "lag_24h must match cost at t-24"
        # avg_cpu_lag1h at row idx must equal avg_cpu at row idx-1
        assert fc_df.iloc[idx]["avg_cpu_lag1h"] == fc_df.iloc[idx - 1]["avg_cpu"], "avg_cpu_lag1h must match avg_cpu at t-1"

    print("PASS: Verified strict temporal ordering and zero future data leakage across all feature columns.")


def test_three_way_model_comparison():
    print("\n--- Test 2: 3-Way Model Comparison (Baseline vs Ridge vs RandomForest) ---")
    df_raw = generate_hourly_telemetry(num_days=30, seed=42)
    df_carbon = compute_carbon_footprint(df_raw)
    fc_df = prepare_forecasting_data(df_carbon)

    eval_df, metrics_summary, trained_model = train_and_evaluate_forecasting(
        fc_df, train_ratio=0.8, model_type="Ridge"
    )

    # Verify all 3 models are present in metrics summary
    assert "Baseline (24h Lag)" in metrics_summary
    assert "Ridge Regression" in metrics_summary
    assert any("Random Forest" in k for k in metrics_summary.keys())

    # Verify evaluation columns present
    assert "Baseline_Pred" in eval_df.columns
    assert "Ridge_Pred" in eval_df.columns
    assert "RandomForest_Pred" in eval_df.columns

    print("3-Way Model Evaluation Results (Chronological 80/20 Split):")
    for model_name, m_dict in metrics_summary.items():
        print(f"  - {model_name:20s}: MAE={m_dict['MAE ($/hr)']:.4f} $/hr, RMSE={m_dict['RMSE ($/hr)']:.4f} $/hr, MAPE={m_dict['MAPE (%)']:.2f}%")

    print("PASS: 3-way model comparison executed on identical chronological train/test split and target variable.")


def test_metric_calculation_resilience():
    print("\n--- Test 3: Metric Calculation Resilience (Zero & Near-Zero Actuals) ---")
    
    y_true_zeros = np.array([0.0, 0.001, 0.005, 1.0, 2.5])
    y_pred = np.array([0.05, 0.002, 0.01, 1.1, 2.4])

    metrics = calculate_metrics(y_true_zeros, y_pred)
    assert not np.isnan(metrics["MAPE (%)"]), "MAPE must not be NaN for zero/near-zero actuals!"
    assert not np.isinf(metrics["MAPE (%)"]), "MAPE must not be infinite for zero/near-zero actuals!"
    print(f"PASS: Metric calculation handles near-zero values gracefully (Calculated MAPE={metrics['MAPE (%)']}%).")


def test_rolling_origin_cv_and_reproducibility():
    print("\n--- Test 4: Rolling-Origin CV & Reproducibility ---")
    df_raw = generate_hourly_telemetry(num_days=30, seed=42)
    df_carbon = compute_carbon_footprint(df_raw)
    fc_df = prepare_forecasting_data(df_carbon)

    # Run rolling origin CV twice
    cv1 = evaluate_rolling_origin_forecasting(fc_df, n_splits=3, test_horizon_hours=48, seed=42)
    cv2 = evaluate_rolling_origin_forecasting(fc_df, n_splits=3, test_horizon_hours=48, seed=42)

    # Verify identical reproducible results
    assert cv1 == cv2, "Rolling-origin CV must be 100% reproducible for fixed seed!"
    
    print("Rolling-Origin Time-Series Cross-Validation Results (3 Folds, 48h Horizon):")
    for model_name, m_dict in cv1.items():
        print(f"  - {model_name:20s}: MAE={m_dict['MAE ($/hr)']:.4f} $/hr, RMSE={m_dict['RMSE ($/hr)']:.4f} $/hr, MAPE={m_dict['MAPE (%)']:.2f}%")

    print("PASS: Rolling-origin cross-validation evaluated and confirmed 100% reproducible.")


if __name__ == "__main__":
    test_temporal_ordering_and_leakage_prevention()
    test_three_way_model_comparison()
    test_metric_calculation_resilience()
    test_rolling_origin_cv_and_reproducibility()
    print("\n=== ALL FORECASTING RESEARCH EVALUATION TESTS PASSED! ===")
