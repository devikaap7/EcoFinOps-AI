"""
Carbon Estimation Engine for EcoFinOps AI.

Calculates estimated energy consumption (kWh) and greenhouse gas emissions (gCO2eq / kgCO2eq)
for cloud workloads based on CPU utilization, hardware TDP/idle power, Power Usage Effectiveness (PUE),
and regional grid carbon intensity coefficients.

IMPORTANT DISCLAIMER:
All carbon emissions figures calculated by this module are software-based ESTIMATES
derived from published hardware specs and grid emission factors. They do NOT represent
direct physical hardware sensor measurements.
"""

import pandas as pd


def compute_carbon_footprint(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes energy consumption (kWh) and estimated carbon emissions (kg CO2e)
    for each hourly record in the dataframe.

    Formulas:
    1. Power (Watts) = Idle_Watts + (TDP_Watts - Idle_Watts) * (CPU_Util / 100)
    2. Energy (kWh) = (Power_Watts * PUE * Usage_Hours) / 1000
    3. Emissions (gCO2e) = Energy_kWh * Grid_Carbon_Intensity_gCO2e_kWh
    4. Emissions (kgCO2e) = Emissions_gCO2e / 1000
    """
    df = df.copy()

    # Calculate estimated power draw in Watts
    power_watts = df["idle_watts"] + (df["tdp_watts"] - df["idle_watts"]) * (df["cpu_utilization_pct"] / 100.0)

    # Calculate total energy in kWh considering data center PUE
    energy_kwh = (power_watts * df["pue"] * df["usage_hours"]) / 1000.0

    # Calculate carbon emissions in gCO2e and kgCO2e
    emissions_gco2e = energy_kwh * df["grid_gco2_kwh"]
    emissions_kgco2e = emissions_gco2e / 1000.0

    if "is_carbon_estimated" in df.columns:
        unmapped_mask = ~df["is_carbon_estimated"].astype(bool)
        power_watts.loc[unmapped_mask] = 0.0
        energy_kwh.loc[unmapped_mask] = 0.0
        emissions_gco2e.loc[unmapped_mask] = 0.0
        emissions_kgco2e.loc[unmapped_mask] = 0.0

    df["power_draw_watts"] = round(power_watts, 2)
    df["energy_consumed_kwh"] = round(energy_kwh, 4)
    df["estimated_gco2e"] = round(emissions_gco2e, 2)
    df["estimated_kgco2e"] = round(emissions_kgco2e, 4)

    return df


def get_carbon_methodology_doc() -> str:
    """
    Returns markdown text explaining the carbon estimation methodology and assumptions.
    """
    return """
    ### 🌿 Carbon Estimation Methodology & Assumptions

    EcoFinOps AI estimates greenhouse gas (GHG) emissions from cloud workloads using the following standard equation:

    $$\\text{Power (W)} = P_{\\text{idle}} + \\left(P_{\\text{max}} - P_{\\text{idle}}\\right) \\times \\frac{\\text{CPU Utilization (\\%)}}{100}$$

    $$\\text{Energy (kWh)} = \\frac{\\text{Power (W)} \\times \\text{PUE} \\times \\text{Duration (hours)}}{1000}$$

    $$\\text{Emissions (kg CO}_2\\text{e)} = \\frac{\\text{Energy (kWh)} \\times \\text{Grid Carbon Intensity (g CO}_2\\text{e/kWh)}}{1000}$$

    #### Hardware & Infrastructure Assumptions:
    * **PUE (Power Usage Effectiveness)**: Ranges between **1.12 - 1.22** across cloud regions (representing data center cooling & power overhead).
    * **TDP (Thermal Design Power)**:
      * `t3.medium` (2 vCPU, 4GB RAM): 25W TDP, 8W Idle
      * `c5.xlarge` (4 vCPU, 8GB RAM): 110W TDP, 35W Idle
      * `r5.2xlarge` (8 vCPU, 64GB RAM): 210W TDP, 60W Idle
      * `g4dn.xlarge` (4 vCPU, 16GB + GPU): 300W TDP, 90W Idle
    * **Grid Carbon Intensities (g CO₂e / kWh)**:
      * **US West (Oregon)**: 120 g CO₂e/kWh (Hydro-dominant)
      * **Europe (Ireland)**: 260 g CO₂e/kWh (Mixed renewable/gas)
      * **US East (N. Virginia)**: 385 g CO₂e/kWh (Grid average)
      * **AP (Mumbai)**: 708 g CO₂e/kWh (Coal-heavy grid)

    > ⚠️ **Disclaimer**: All carbon metrics reported by EcoFinOps AI are **software-based estimates** calculated for analytical and optimization purposes. They are not direct physical meter readings.
    """
