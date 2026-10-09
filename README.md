# 🌿 EcoFinOps AI: Agentic Cloud Cost and Carbon Optimization Platform (V1)

**EcoFinOps AI** is an open-source, reproducible platform designed to optimize cloud infrastructure financial spend ($) and environmental carbon footprint ($\text{kg CO}_2\text{e}$). Version 1 runs completely locally with **zero paid cloud services or API key requirements**, making it ideal for learning, experimentation, and academic demonstration.

---

## 📁 Project Structure & File Explanations

```
EcoFinOps-AI/
├── .venv/                      # Python Virtual Environment
├── requirements.txt            # Locked project dependencies (pandas, scikit-learn, streamlit, plotly)
├── README.md                   # Project documentation & execution guide
├── app.py                      # Main Streamlit web application & UI dashboard
├── src/
│   ├── __init__.py             # Python package marker
│   ├── data_generator.py       # Generates realistic synthetic hourly workload & billing telemetry
│   ├── carbon_engine.py        # PUE, TDP & regional grid carbon intensity calculation engine
│   ├── forecasting.py          # Chronological ML cost forecasting (ML vs 24h Seasonal Baseline)
│   ├── anomaly_detector.py     # Isolation Forest & rule-based idle waste / cost spike detector
│   └── optimizer.py            # Tri-strategy workload scheduler (Baseline vs Cost-Aware vs Carbon-Aware)
└── scratch/
    └── test_pipeline.py        # Automated test verification script
```

### Module Breakdown
* **[`app.py`](file:///c:/Users/DEVIKA/Downloads/EcoFinOps-AI/app.py)**: Interactive multi-tab Streamlit dashboard presenting executive KPIs, forecasting charts, anomaly alerts, optimization scenarios, and methodology details.
* **[`src/data_generator.py`](file:///c:/Users/DEVIKA/Downloads/EcoFinOps-AI/src/data_generator.py)**: Generates hourly multi-region (`us-east-1`, `us-west-2`, `eu-west-1`, `ap-south-1`) cloud usage, CPU/RAM utilization curves, diurnal patterns, and workload metadata without external APIs.
* **[`src/carbon_engine.py`](file:///c:/Users/DEVIKA/Downloads/EcoFinOps-AI/src/carbon_engine.py)**: Computes estimated energy consumption ($E_{\text{kWh}}$) and greenhouse gas emissions ($\text{kg CO}_2\text{e}$) using hardware Thermal Design Power (TDP), Data Center PUE (1.12–1.22), and regional grid intensity factors (120–708 $\text{gCO}_2\text{e/kWh}$).
* **[`src/forecasting.py`](file:///c:/Users/DEVIKA/Downloads/EcoFinOps-AI/src/forecasting.py)**: Evaluates Ridge Regression and Random Forest models against a 24-hour seasonal lag baseline. Uses strict **chronological train-test splitting (80/20)** to prevent temporal data leakage.
* **[`src/anomaly_detector.py`](file:///c:/Users/DEVIKA/Downloads/EcoFinOps-AI/src/anomaly_detector.py)**: Combines Isolation Forest with Z-score thresholding to flag cost spikes and zombie/idle compute instances ($\text{CPU} < 10\%$).
* **[`src/optimizer.py`](file:///c:/Users/DEVIKA/Downloads/EcoFinOps-AI/src/optimizer.py)**: Shifts flexible batch workloads within arrival and deadline windows across 24-hour grid pricing and carbon intensity profiles.

---

## 🚀 Step-by-Step Instructions to Run on Windows

Follow these exact steps in **PowerShell**:

### Step 1: Open Terminal in Project Folder
```powershell
cd c:\Users\DEVIKA\Downloads\EcoFinOps-AI
```

### Step 2: Ensure Python 3 Dependencies are Installed
If running for the first time, install the required packages:
```powershell
py -3 -m pip install -r requirements.txt
```

### Step 3: Run Pipeline Verification Test
Run the backend test script to verify ML pipelines, carbon engine, and data generation:
```powershell
$env:PYTHONPATH="."; py -3 scratch\test_pipeline.py
```

### Step 4: Launch the Streamlit Dashboard
Launch the interactive Streamlit web dashboard:
```powershell
$env:PYTHONPATH="."; py -3 -m streamlit run app.py
```

Open your browser at: **`http://localhost:8501`**

---

## 🔬 Mathematical & Carbon Accounting Assumptions

1. **Hardware Power Equation**:
   $$\text{Power (W)} = P_{\text{idle}} + (P_{\text{TDP}} - P_{\text{idle}}) \times \frac{\text{CPU}\%}{100}$$

2. **Energy Consumption**:
   $$\text{Energy (kWh)} = \frac{\text{Power (W)} \times \text{PUE} \times \text{Hours}}{1000}$$

3. **Carbon Footprint**:
   $$\text{Emissions (kg CO}_2\text{e)} = \frac{\text{Energy (kWh)} \times \text{Grid Intensity (g CO}_2\text{e/kWh)}}{1000}$$

> **Software Disclaimer**: All carbon values reported are software-based estimates derived from published PUE and grid intensity literature, not direct physical meter hardware sensors.

---

## 📊 Experimental Results (V1 Baseline)

| Strategy | Total Cost ($) | Total Carbon (kg CO₂e) | Deadline Compliance | Carbon Reduction (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Immediate)** | $7.85 | 0.0508 | 100.0% | 0.0% |
| **Cost-Aware Optimizer** | $7.72 | 0.0500 | 100.0% | 1.6% |
| **Carbon-Aware Optimizer** | $7.98 | 0.0409 | 100.0% | **19.4%** |
