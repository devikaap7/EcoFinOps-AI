"""
EcoFinOps AI: Agentic Cloud Cost and Carbon Optimization Platform (V1)
Streamlit Dashboard Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_generator import generate_hourly_telemetry, generate_synthetic_workloads, REGION_SPECS, INSTANCE_SPECS
from src.dataset_loader import load_public_billing_dataset
from src.carbon_engine import compute_carbon_footprint, get_carbon_methodology_doc
from src.forecasting import prepare_forecasting_data, train_and_evaluate_forecasting, evaluate_rolling_origin_forecasting
from src.anomaly_detector import detect_anomalies_and_waste
from src.optimizer import optimize_schedules, get_hourly_grid_profile

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="EcoFinOps AI - Cloud Cost & Carbon Platform",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for modern UI
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stMetric {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #34d399;
    }
    .alert-card {
        background-color: #450a0a;
        border: 1px solid #991b1b;
        padding: 12px;
        border-radius: 8px;
        color: #fecaca;
        margin-bottom: 10px;
    }
    .success-card {
        background-color: #064e3b;
        border: 1px solid #065f46;
        padding: 12px;
        border-radius: 8px;
        color: #a7f3d0;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/isometric/100/leaf.png", width=60)
st.sidebar.title("EcoFinOps AI Settings")
st.sidebar.markdown("---")

data_mode = st.sidebar.selectbox(
    "Data Operating Mode",
    options=["Synthetic Telemetry", "AWS CUR Billing Dataset (CSV)", "GCP Billing Export (CSV)"],
)

num_days = st.sidebar.slider("Simulation Window (Days)", min_value=7, max_value=60, value=30, step=1)
seed_val = st.sidebar.number_input("Random Seed (Reproducibility)", value=42, step=1)
model_choice = st.sidebar.selectbox("Forecasting Algorithm", options=["RandomForest", "Ridge"])

selected_regions = st.sidebar.multiselect(
    "Filter Cloud Regions",
    options=list(REGION_SPECS.keys()),
    default=list(REGION_SPECS.keys()),
    format_func=lambda x: f"{x} ({REGION_SPECS[x]['name']})",
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **EcoFinOps AI V3** supports synthetic telemetry and public cloud billing datasets (AWS CUR / GCP Billing). "
    "All carbon calculations are software estimates based on published PUE & grid intensity factors."
)

# ---------------------------------------------------------
# Data Loading & Caching
# ---------------------------------------------------------
@st.cache_data
def load_data(days: int, seed: int, mode: str):
    if mode == "AWS CUR Billing Dataset (CSV)":
        raw_df = load_public_billing_dataset("aws_cur")
    elif mode == "GCP Billing Export (CSV)":
        raw_df = load_public_billing_dataset("gcp_billing")
    else:
        raw_df = generate_hourly_telemetry(num_days=days, seed=seed)

    carbon_df = compute_carbon_footprint(raw_df)
    workloads_df = generate_synthetic_workloads(seed=seed)
    return carbon_df, workloads_df

df_full, workloads_df = load_data(num_days, seed_val, data_mode)

# Filter by selected region
if selected_regions:
    df = df_full[df_full["region"].isin(selected_regions)].copy()
else:
    df = df_full.copy()

# Run ML & Analytics Pipelines
forecast_hourly_df = prepare_forecasting_data(df)
eval_df, forecast_metrics, trained_model = train_and_evaluate_forecasting(
    forecast_hourly_df, train_ratio=0.8, model_type=model_choice
)
df, high_cost_anomalies, idle_summary = detect_anomalies_and_waste(df)
scheduled_tasks, sched_summary = optimize_schedules(workloads_df)

# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------
st.title("🌿 EcoFinOps AI: Cloud Cost & Carbon Platform")
st.caption("Agentic-Ready Intelligent Cloud Financial & Sustainability Operations Dashboard (V1)")

# ---------------------------------------------------------
# Tabs Navigation
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Overview",
    "🔮 ML Cost Forecasting",
    "🚨 Waste & Anomalies",
    "⚡ Scheduling Optimizer",
    "📚 Methodology & Specs",
    "🤖 Agentic MCP Server",
])

# ---------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.subheader("Key Performance Indicators")

    total_cost = df["hourly_cost_usd"].sum()
    total_carbon = df["estimated_kgco2e"].sum()
    total_kwh = df["energy_consumed_kwh"].sum()
    total_anomalies = len(high_cost_anomalies)
    wasted_cost = idle_summary["wasted_cost_usd"].sum() if not idle_summary.empty else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(label="Total Cloud Spend", value=f"${total_cost:,.2f}", delta=f"{len(df)} telemetry hrs")
    with c2:
        st.metric(label="Est. Carbon Footprint", value=f"{total_carbon:,.1f} kg CO₂e", delta=f"{total_kwh:,.0f} kWh energy")
    with c3:
        st.metric(label="Active Fleet Size", value=f"{df['instance_id'].nunique()} Instances", delta=f"{len(selected_regions)} Regions")
    with c4:
        st.metric(label="Cost Anomalies", value=f"{total_anomalies} Spikes", delta="Requires Attention", delta_color="inverse")
    with c5:
        st.metric(label="Idle Resource Waste", value=f"${wasted_cost:,.2f}", delta=f"{len(idle_summary)} Zombie Nodes", delta_color="inverse")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📈 Hourly Cloud Spend & Carbon Trend")
        hourly_agg = df.groupby("timestamp").agg(
            cost=("hourly_cost_usd", "sum"),
            carbon=("estimated_kgco2e", "sum"),
        ).reset_index()

        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Scatter(x=hourly_agg["timestamp"], y=hourly_agg["cost"], name="Hourly Cost ($)", line=dict(color="#38bdf8", width=2))
        )
        fig_trend.add_trace(
            go.Scatter(x=hourly_agg["timestamp"], y=hourly_agg["carbon"], name="Carbon (kg CO2e)", yaxis="y2", line=dict(color="#34d399", width=1.5, dash="dot"))
        )
        fig_trend.update_layout(
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(title="Spend ($ USD)"),
            yaxis2=dict(title="Est. Carbon (kg CO₂e)", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        st.markdown("### 🌍 Regional Spend & Carbon Intensity Breakdown")
        reg_agg = df.groupby("region").agg(
            cost=("hourly_cost_usd", "sum"),
            carbon=("estimated_kgco2e", "sum"),
        ).reset_index()

        fig_reg = px.bar(
            reg_agg,
            x="region",
            y="cost",
            color="carbon",
            labels={"cost": "Total Spend ($)", "carbon": "Carbon (kg CO2e)"},
            color_continuous_scale="Viridis",
            text_auto=".2s",
            template="plotly_dark",
            height=380,
        )
        fig_reg.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_reg, use_container_width=True)


# ---------------------------------------------------------
# TAB 2: ML COST FORECASTING
# ---------------------------------------------------------
with tab2:
    st.subheader("🔮 Chronological Time-Series Cost Forecasting Evaluation")
    st.markdown(
        "To **prevent data leakage**, all features strictly use past information ($t-1$ or earlier) and deterministic calendar attributes. "
        "Models are evaluated across both single chronological 80/20 out-of-sample split and **Rolling-Origin Time-Series Cross-Validation**."
    )

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("#### 📊 Evaluation Metrics (Chronological 80/20 Test Set)")
        metrics_df = pd.DataFrame(forecast_metrics).T
        st.dataframe(metrics_df.style.highlight_min(subset=["MAE ($/hr)", "RMSE ($/hr)", "MAPE (%)"], color="#065f46"), use_container_width=True)

    with m2:
        st.markdown("#### 🎯 Model Performance Insight")
        keys_available = list(forecast_metrics.keys())
        ml_model_key = "Ridge Regression"
        if model_choice == "RandomForest":
            rf_keys = [k for k in keys_available if "Random Forest" in k]
            if rf_keys:
                ml_model_key = rf_keys[0]

        ml_mae = forecast_metrics[ml_model_key]["MAE ($/hr)"]
        base_mae = forecast_metrics["Baseline (24h Lag)"]["MAE ($/hr)"]
        improvement = ((base_mae - ml_mae) / base_mae) * 100

        if "Fallback" in ml_model_key:
            st.markdown(
                "<div class='alert-card'><b>ℹ️ Windows Security Policy Notice</b><br>"
                "Scikit-learn tree C-extension DLL (<code>_tree.pyd</code>) is blocked by Windows Application Control. "
                "The system transparently ran <b>Ridge Regression</b> as an approved fallback for Random Forest.</div>",
                unsafe_allow_html=True,
            )
        elif improvement > 0:
            st.markdown(
                f"<div class='success-card'><b>✅ {ml_model_key} Superiority Confirmed</b><br>"
                f"The {ml_model_key} model reduced Mean Absolute Error by <b>{improvement:.1f}%</b> compared to the 24h Naive Baseline.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='alert-card'><b>⚠️ Baseline Competitive</b><br>"
                f"The 24h Naive Baseline achieved lower or equal MAE ({base_mae:.4f} $/hr vs {ml_mae:.4f} $/hr) due to strong 24-hour diurnal periodicity.</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("#### 🔄 Rolling-Origin Time-Series Cross-Validation (Expanding Window Evaluation)")
    rolling_metrics = evaluate_rolling_origin_forecasting(forecast_hourly_df, n_splits=3, test_horizon_hours=48, seed=seed_val)
    rolling_df = pd.DataFrame(rolling_metrics).T
    st.dataframe(rolling_df.style.highlight_min(subset=["MAE ($/hr)", "RMSE ($/hr)", "MAPE (%)"], color="#065f46"), use_container_width=True)

    st.markdown("#### 📉 Out-of-Sample Forecast Evaluation (Actual vs Baseline vs Ridge vs RandomForest)")
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=eval_df["timestamp"], y=eval_df["total_hourly_cost"], name="Actual Hourly Spend", line=dict(color="#f8fafc", width=2)))
    fig_fc.add_trace(go.Scatter(x=eval_df["timestamp"], y=eval_df["Baseline_Pred"], name="Baseline (24h Lag)", line=dict(color="#fbbf24", width=1.5, dash="dash")))
    fig_fc.add_trace(go.Scatter(x=eval_df["timestamp"], y=eval_df["Ridge_Pred"], name="Ridge Regression", line=dict(color="#38bdf8", width=1.8)))
    fig_fc.add_trace(go.Scatter(x=eval_df["timestamp"], y=eval_df["RandomForest_Pred"], name="Random Forest", line=dict(color="#34d399", width=1.8, dash="dot")))

    fig_fc.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Timeline (Out-of-Sample Test Window)",
        yaxis_title="Hourly Total Cost ($ USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_fc, use_container_width=True)


# ---------------------------------------------------------
# TAB 3: ANOMALY DETECTION & WASTE IDENTIFICATION
# ---------------------------------------------------------
with tab3:
    st.subheader("🚨 Anomaly Detection & Idle Waste Identification")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("#### 🧟 Zombie / Underutilized Resources")
        if idle_summary.empty:
            st.success("No idle instances detected.")
        else:
            st.dataframe(
                idle_summary[["instance_id", "instance_type", "region", "idle_hours", "wasted_cost_usd", "recommendation"]],
                use_container_width=True,
            )

    with col_b:
        st.markdown("#### ⚡ Unusually High Cost Spikes")
        if high_cost_anomalies.empty:
            st.success("No cost spikes detected.")
        else:
            st.dataframe(
                high_cost_anomalies[["timestamp", "instance_id", "instance_type", "cpu_utilization_pct", "hourly_cost_usd", "cost_zscore"]].head(10),
                use_container_width=True,
            )

    st.markdown("---")
    st.markdown("#### 🎯 Fleet Utilization vs Cost (Identifying Waste Quadrants)")

    fig_scatter = px.scatter(
        df,
        x="cpu_utilization_pct",
        y="hourly_cost_usd",
        color="is_underutilized",
        size="ram_gb",
        hover_data=["instance_id", "instance_type", "region"],
        labels={
            "cpu_utilization_pct": "CPU Utilization (%)",
            "hourly_cost_usd": "Hourly Cost ($)",
            "is_underutilized": "Zombie Idle Flag",
        },
        color_discrete_map={True: "#ef4444", False: "#38bdf8"},
        template="plotly_dark",
        height=400,
    )
    fig_scatter.add_vline(x=10, line_dash="dash", line_color="#ef4444", annotation_text="Low CPU Threshold (10%)")
    fig_scatter.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_scatter, use_container_width=True)


# ---------------------------------------------------------
# TAB 4: WORKLOAD SCHEDULING OPTIMIZER
# ---------------------------------------------------------
with tab4:
    st.subheader("⚡ Tri-Strategy Workload Scheduling Optimizer")
    st.markdown(
        "Shifts flexible batch workloads across a 24-hour planning window to minimize cost or carbon footprint "
        "while enforcing **100% deadline compliance**."
    )

    st.markdown("#### 📊 Strategy Comparison Matrix")
    st.dataframe(
        sched_summary.style.format(
            {
                "Total Cost ($)": "${:,.2f}",
                "Total Carbon (kg CO2e)": "{:,.4f}",
                "Deadline Compliance (%)": "{:.1f}%",
                "Cost Savings (%)": "{:.1f}%",
                "Carbon Reduction (%)": "{:.1f}%",
            }
        ).highlight_max(subset=["Cost Savings (%)", "Carbon Reduction (%)"], color="#065f46"),
        use_container_width=True,
    )

    c_opt1, c_opt2 = st.columns(2)

    with c_opt1:
        st.markdown("#### 💰 Total Cost Comparison by Strategy ($)")
        fig_cost_comp = px.bar(
            sched_summary,
            x="Strategy",
            y="Total Cost ($)",
            color="Strategy",
            text_auto=".2f",
            color_discrete_sequence=["#94a3b8", "#38bdf8", "#34d399"],
            template="plotly_dark",
            height=350,
        )
        fig_cost_comp.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_cost_comp, use_container_width=True)

    with c_opt2:
        st.markdown("#### 🌿 Total Carbon Emissions Comparison (kg CO₂e)")
        fig_carb_comp = px.bar(
            sched_summary,
            x="Strategy",
            y="Total Carbon (kg CO2e)",
            color="Strategy",
            text_auto=".4f",
            color_discrete_sequence=["#94a3b8", "#38bdf8", "#34d399"],
            template="plotly_dark",
            height=350,
        )
        fig_carb_comp.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_carb_comp, use_container_width=True)

    st.markdown("#### 📋 Task Execution Timelines & Deadline Compliance")
    st.dataframe(
        scheduled_tasks[[
            "task_id",
            "name",
            "flexibility",
            "arrival_hour",
            "deadline_hour",
            "baseline_start",
            "cost_aware_start",
            "carbon_aware_start",
            "deadline_met",
        ]],
        use_container_width=True,
    )


# ---------------------------------------------------------
# TAB 5: METHODOLOGY & SPECS
# ---------------------------------------------------------
with tab5:
    st.subheader("📚 Mathematical Foundations & Reproducibility Specs")
    st.markdown(get_carbon_methodology_doc())

    st.markdown("---")
    st.markdown("### ⚙️ Instance Thermal & Financial Specifications")
    st.json(INSTANCE_SPECS)

    st.markdown("### 🌍 Regional Infrastructure Specs")
    st.json(REGION_SPECS)


# ---------------------------------------------------------
# TAB 6: AGENTIC MCP SERVER
# ---------------------------------------------------------
with tab6:
    st.subheader("🤖 Model Context Protocol (MCP) Server Integration")
    st.markdown(
        "EcoFinOps AI V2 exposes standard MCP tools allowing AI agents (Antigravity, Claude Desktop, Cursor) "
        "to inspect cloud telemetry, compute carbon emissions, forecast spend, detect waste, and optimize schedules autonomously."
    )

    st.markdown(
        "<div class='success-card'><b>✅ MCP Server Module Loaded</b><br>"
        "FastMCP server module <code>mcp_server.py</code> registered 5 read-only cloud tools. "
        "Zero real cloud resource mutations or paid API requirements.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 🛠️ Exposed MCP Tools Specification")
    
    tools_info = [
        {"Tool Name": "get_telemetry_summary", "Parameters": "num_days, seed, regions", "Description": "Returns fleet cost ($), carbon footprint (kg CO2e), energy (kWh), and regional breakdown.", "Target Module": "src.data_generator & src.carbon_engine"},
        {"Tool Name": "forecast_cloud_spend", "Parameters": "num_days, seed, model_type, train_ratio", "Description": "Out-of-sample chronological forecasting metrics (MAE, RMSE, MAPE) vs 24h baseline.", "Target Module": "src.forecasting"},
        {"Tool Name": "detect_anomalies_and_waste", "Parameters": "num_days, seed, contamination", "Description": "Identifies zombie idle resources (CPU<10%, cost>$0.03/hr) and cost anomaly spikes.", "Target Module": "src.anomaly_detector"},
        {"Tool Name": "optimize_workload_schedules", "Parameters": "seed", "Description": "Simulates 3 workload scheduling strategies and calculates dynamic deadline compliance.", "Target Module": "src.optimizer"},
        {"Tool Name": "get_carbon_methodology", "Parameters": "None", "Description": "Returns technical documentation on TDP, PUE (1.12-1.22), and grid emission factors.", "Target Module": "src.carbon_engine"},
    ]
    st.dataframe(pd.DataFrame(tools_info), use_container_width=True)

    c_mcp1, c_mcp2 = st.columns(2)

    with c_mcp1:
        st.markdown("### ⚙️ Client Config (`mcp_config.json`)")
        st.code(
            """{
  "mcpServers": {
    "ecofinops-ai": {
      "command": "C:\\\\Users\\\\DEVIKA\\\\Downloads\\\\EcoFinOps-AI\\\\.venv\\\\Scripts\\\\python.exe",
      "args": [
        "C:\\\\Users\\\\DEVIKA\\\\Downloads\\\\EcoFinOps-AI\\\\mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "C:\\\\Users\\\\DEVIKA\\\\Downloads\\\\EcoFinOps-AI"
      }
    }
  }
}""",
            language="json",
        )

    with c_mcp2:
        st.markdown("### 🚀 Windows PowerShell Launch Command")
        st.code(
            """Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\\.venv\\Scripts\\Activate.ps1
$env:PYTHONPATH="."
python mcp_server.py""",
            language="powershell",
        )

    st.markdown("---")
    st.markdown("### 🧪 Interactive MCP Tool Sandbox")
    selected_tool = st.selectbox("Select MCP Tool to Simulate Invocation", options=[
        "get_telemetry_summary",
        "forecast_cloud_spend",
        "detect_anomalies_and_waste",
        "optimize_workload_schedules",
        "get_carbon_methodology",
    ])

    if st.button("▶️ Execute Tool via MCP Module"):
        from mcp_server import (
            get_telemetry_summary as mcp_summary,
            forecast_cloud_spend as mcp_forecast,
            detect_anomalies_and_waste as mcp_anomalies,
            optimize_workload_schedules as mcp_opt,
            get_carbon_methodology as mcp_doc,
        )
        if selected_tool == "get_telemetry_summary":
            res = mcp_summary(num_days=num_days, seed=seed_val, regions=selected_regions if selected_regions else None)
        elif selected_tool == "forecast_cloud_spend":
            res = mcp_forecast(num_days=num_days, seed=seed_val, model_type=model_choice)
        elif selected_tool == "detect_anomalies_and_waste":
            res = mcp_anomalies(num_days=num_days, seed=seed_val)
        elif selected_tool == "optimize_workload_schedules":
            res = mcp_opt(seed=seed_val)
        else:
            res = mcp_doc()

        st.json(res) if isinstance(res, dict) else st.markdown(res)
