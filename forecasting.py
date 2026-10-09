"""
Time-Series Cost Forecasting Model for EcoFinOps AI.

Features:
- Time-aware chronological train-test split (80% train, 20% test) to prevent data leakage.
- Comparison between a Simple Naive Baseline (24-hour seasonal lag) and ML Model (Ridge/RandomForest).
- Evaluation metrics: MAE, RMSE, MAPE.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare_forecasting_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates telemetry to hourly total cloud spend and engineers time-series features.
    """
    # Aggregate total spend per hour across all instances
    hourly_df = (
        df.groupby("timestamp")
        .agg(
            total_hourly_cost=("hourly_cost_usd", "sum"),
            total_kgco2e=("estimated_kgco2e", "sum"),
            avg_cpu=("cpu_utilization_pct", "mean"),
            active_instances=("instance_id", "nunique"),
        )
        .reset_index()
        .sort_values("timestamp")
    )

    # Engineer time-series features (Strictly past-only / zero future leakage)
    hourly_df["hour"] = hourly_df["timestamp"].dt.hour
    hourly_df["day_of_week"] = hourly_df["timestamp"].dt.dayofweek
    hourly_df["is_weekend"] = (hourly_df["day_of_week"] >= 5).astype(int)

    # Lags & rolling windows (shifted by .shift(1) to use data prior to hour t)
    hourly_df["lag_1h"] = hourly_df["total_hourly_cost"].shift(1)
    hourly_df["lag_24h"] = hourly_df["total_hourly_cost"].shift(24)
    hourly_df["rolling_mean_6h"] = hourly_df["total_hourly_cost"].shift(1).rolling(6).mean()
    hourly_df["rolling_mean_24h"] = hourly_df["total_hourly_cost"].shift(1).rolling(24).mean()
    hourly_df["avg_cpu_lag1h"] = hourly_df["avg_cpu"].shift(1)
    hourly_df["active_instances_lag1h"] = hourly_df["active_instances"].shift(1)

    # Drop NaNs created by lagging
    hourly_df = hourly_df.dropna().reset_index(drop=True)

    return hourly_df


FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "lag_1h",
    "lag_24h",
    "rolling_mean_6h",
    "rolling_mean_24h",
    "avg_cpu_lag1h",
    "active_instances_lag1h",
]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes MAE, RMSE, and robust clipped-epsilon MAPE.
    Denominator clipping (at $0.01/hr) prevents division by zero or near-zero distortion.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = np.maximum(np.abs(y_true), 0.01)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    return {
        "MAE ($/hr)": round(float(mae), 4),
        "RMSE ($/hr)": round(float(rmse), 4),
        "MAPE (%)": round(float(mape), 2),
    }


def train_and_evaluate_forecasting(
    hourly_df: pd.DataFrame, train_ratio: float = 0.8, model_type: str = "Ridge"
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], object]:
    """
    Chronologically splits data into Train and Test sets to evaluate Baseline vs Ridge vs RandomForest.
    Prevents temporal data leakage by using only lagged and deterministic calendar features.
    Returns 3-way evaluation metrics for all models.
    """
    target_col = "total_hourly_cost"
    split_idx = int(len(hourly_df) * train_ratio)

    train_df = hourly_df.iloc[:split_idx].copy()
    test_df = hourly_df.iloc[split_idx:].copy()

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[target_col]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[target_col]

    # Baseline Model: 24-hour seasonal lag
    baseline_preds = test_df["lag_24h"].values

    # Ridge Model
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    ridge_preds = ridge_model.predict(X_test)

    # Random Forest Model (with fallback for OS DLL security policies)
    try:
        from sklearn.ensemble import RandomForestRegressor
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)
        rf_available = True
    except (ImportError, Exception):
        rf_model = ridge_model
        rf_preds = ridge_preds
        rf_available = False

    # Calculate metrics for all 3 models
    baseline_metrics = calculate_metrics(y_test, baseline_preds)
    ridge_metrics = calculate_metrics(y_test, ridge_preds)
    
    rf_label = "Random Forest" if rf_available else "Random Forest (Fallback: Ridge)"
    rf_metrics = calculate_metrics(y_test, rf_preds)

    metrics_summary = {
        "Baseline (24h Lag)": baseline_metrics,
        "Ridge Regression": ridge_metrics,
        rf_label: rf_metrics,
    }

    # Store predictions for visualization
    eval_df = test_df[["timestamp", target_col]].copy()
    eval_df["Baseline_Pred"] = baseline_preds
    eval_df["Ridge_Pred"] = ridge_preds
    eval_df["RandomForest_Pred"] = rf_preds
    eval_df["ML_Model_Pred"] = ridge_preds if (model_type == "Ridge" or not rf_available) else rf_preds
    eval_df["rf_available"] = rf_available

    selected_model = ridge_model if (model_type == "Ridge" or not rf_available) else rf_model
    return eval_df, metrics_summary, selected_model


def evaluate_rolling_origin_forecasting(
    hourly_df: pd.DataFrame, n_splits: int = 3, test_horizon_hours: int = 48, seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Performs Rolling-Origin Time-Series Cross-Validation (Expanding Window Evaluation).
    Evaluates Baseline, Ridge, and Random Forest across multiple consecutive time horizons.
    """
    total_len = len(hourly_df)
    min_train = int(total_len * 0.5)
    step = (total_len - min_train - test_horizon_hours) // max(n_splits - 1, 1)
    
    baseline_maes, baseline_rmses, baseline_mapes = [], [], []
    ridge_maes, ridge_rmses, ridge_mapes = [], [], []
    rf_maes, rf_rmses, rf_mapes = [], [], []
    rf_available = True

    for fold in range(n_splits):
        train_end = min_train + fold * step
        test_end = min(train_end + test_horizon_hours, total_len)

        train_fold = hourly_df.iloc[:train_end]
        test_fold = hourly_df.iloc[train_end:test_end]
        
        if len(test_fold) < 12:
            continue

        X_tr = train_fold[FEATURE_COLS]
        y_tr = train_fold["total_hourly_cost"]
        X_te = test_fold[FEATURE_COLS]
        y_te = test_fold["total_hourly_cost"].values

        base_p = test_fold["lag_24h"].values
        
        r_mod = Ridge(alpha=1.0)
        r_mod.fit(X_tr, y_tr)
        r_p = r_mod.predict(X_te)

        try:
            from sklearn.ensemble import RandomForestRegressor
            rf_mod = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed)
            rf_mod.fit(X_tr, y_tr)
            rf_p = rf_mod.predict(X_te)
        except (ImportError, Exception):
            rf_p = r_p
            rf_available = False

        b_m = calculate_metrics(y_te, base_p)
        r_m = calculate_metrics(y_te, r_p)
        rf_m = calculate_metrics(y_te, rf_p)

        baseline_maes.append(b_m["MAE ($/hr)"])
        baseline_rmses.append(b_m["RMSE ($/hr)"])
        baseline_mapes.append(b_m["MAPE (%)"])

        ridge_maes.append(r_m["MAE ($/hr)"])
        ridge_rmses.append(r_m["RMSE ($/hr)"])
        ridge_mapes.append(r_m["MAPE (%)"])

        rf_maes.append(rf_m["MAE ($/hr)"])
        rf_rmses.append(rf_m["RMSE ($/hr)"])
        rf_mapes.append(rf_m["MAPE (%)"])

    rf_label = "Random Forest" if rf_available else "Random Forest (Fallback: Ridge)"
    return {
        "Baseline (24h Lag)": {
            "MAE ($/hr)": round(float(np.mean(baseline_maes)), 4),
            "RMSE ($/hr)": round(float(np.mean(baseline_rmses)), 4),
            "MAPE (%)": round(float(np.mean(baseline_mapes)), 2),
        },
        "Ridge Regression": {
            "MAE ($/hr)": round(float(np.mean(ridge_maes)), 4),
            "RMSE ($/hr)": round(float(np.mean(ridge_rmses)), 4),
            "MAPE (%)": round(float(np.mean(ridge_mapes)), 2),
        },
        rf_label: {
            "MAE ($/hr)": round(float(np.mean(rf_maes)), 4),
            "RMSE ($/hr)": round(float(np.mean(rf_rmses)), 4),
            "MAPE (%)": round(float(np.mean(rf_mapes)), 2),
        },
    }
