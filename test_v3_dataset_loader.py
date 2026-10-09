"""
V3 Public Cloud Billing Dataset Loader & MCP Data Mode Automated Test Suite.
Verifies AWS CUR and GCP Billing Export parsing, schema validation, currency check,
explicit field mappings, carbon disclosure rules, and MCP tool multi-mode support.
"""

import os
import sys
import tempfile
import pandas as pd

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_loader import load_public_billing_dataset
from src.carbon_engine import compute_carbon_footprint
from mcp_server import (
    get_telemetry_summary,
    forecast_cloud_spend,
    detect_anomalies_and_waste,
    optimize_workload_schedules,
)


def test_aws_cur_ingestion_and_carbon():
    print("\n--- Test 1: AWS CUR Ingestion & Carbon Estimation ---")
    df = load_public_billing_dataset("aws_cur")
    
    assert len(df) > 0, "AWS CUR dataset should not be empty."
    expected_cols = [
        "timestamp", "instance_id", "instance_type", "region",
        "hourly_cost_usd", "usage_hours", "is_carbon_estimated", "unmapped_reason"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column '{col}' in AWS CUR dataframe."

    # Compute carbon footprint
    df_carbon = compute_carbon_footprint(df)
    
    # Check that unmapped line items have 0 kg CO2e
    unmapped = df_carbon[~df_carbon["is_carbon_estimated"]]
    assert len(unmapped) > 0, "Expected at least one unmapped line item in sample AWS CUR dataset."
    for _, row in unmapped.iterrows():
        assert row["estimated_kgco2e"] == 0.0, f"Unmapped row should have 0.0 kg CO2e, got {row['estimated_kgco2e']}"
        assert row["unmapped_reason"] is not None, "Unmapped row must disclose unmapped_reason."

    # Check mapped compute items have positive carbon
    mapped = df_carbon[df_carbon["is_carbon_estimated"]]
    assert len(mapped) > 0, "Expected mapped compute items in sample AWS CUR dataset."
    assert (mapped["estimated_kgco2e"] > 0).all(), "Mapped compute items should have positive carbon estimates."
    
    print(f"PASS: AWS CUR ingestion test passed. Ingested {len(df)} records ({len(mapped)} mapped, {len(unmapped)} disclosed unmapped).")


def test_gcp_billing_ingestion_and_mapping():
    print("\n--- Test 2: GCP Billing Export Ingestion & Mapping ---")
    df = load_public_billing_dataset("gcp_billing")
    
    assert len(df) > 0, "GCP Billing dataset should not be empty."
    
    df_carbon = compute_carbon_footprint(df)
    unmapped = df_carbon[~df_carbon["is_carbon_estimated"]]
    mapped = df_carbon[df_carbon["is_carbon_estimated"]]

    assert len(unmapped) > 0, "Storage SKUs in GCP billing export should be marked unmapped."
    assert (unmapped["estimated_kgco2e"] == 0.0).all(), "Unmapped storage SKUs must have 0.0 kg CO2e."
    assert (mapped["estimated_kgco2e"] > 0).all(), "Mapped GCP compute SKUs must have positive carbon."
    
    print(f"PASS: GCP Billing ingestion test passed. Ingested {len(df)} records ({len(mapped)} mapped, {len(unmapped)} disclosed unmapped).")


def test_schema_validation_and_currency_check():
    print("\n--- Test 3: Schema Validation & Non-USD Currency Rejection ---")
    
    # 1. Invalid dataset_type
    try:
        load_public_billing_dataset("invalid_type")
        assert False, "Should have raised ValueError for invalid dataset_type"
    except ValueError as e:
        assert "Unsupported dataset_type" in str(e)

    # 2. Non-existent file
    try:
        load_public_billing_dataset("aws_cur", custom_file_path="non_existent_file.csv")
        assert False, "Should have raised FileNotFoundError for non-existent file"
    except FileNotFoundError:
        pass

    # 3. Non-USD currency rejection test
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
        tmp.write("lineItem/UsageStartDate,lineItem/UnblendedCost,lineItem/UsageAmount,lineItem/CurrencyCode\n")
        tmp.write("2026-03-01 00:00:00,1.50,1.0,EUR\n")
        tmp_path = tmp.name

    try:
        try:
            load_public_billing_dataset("aws_cur", custom_file_path=tmp_path)
            assert False, "Should have raised ValueError for EUR currency"
        except ValueError as e:
            assert "Non-USD currency detected" in str(e)
            print("PASS: Correctly rejected non-USD (EUR) currency with explicit ValueError.")
    finally:
        os.remove(tmp_path)


def test_mcp_tools_with_public_data_modes():
    print("\n--- Test 4: MCP Tools Multi-Data-Mode Support ---")

    # Test get_telemetry_summary with aws_cur and gcp_billing
    aws_sum = get_telemetry_summary(data_mode="aws_cur")
    assert aws_sum["data_mode"] == "aws_cur"
    assert aws_sum["total_cost_usd"] > 0
    assert aws_sum["unmapped_carbon_records"] >= 1

    gcp_sum = get_telemetry_summary(data_mode="gcp_billing")
    assert gcp_sum["data_mode"] == "gcp_billing"
    assert gcp_sum["total_cost_usd"] > 0

    # Test forecast_cloud_spend with aws_cur
    aws_fc = forecast_cloud_spend(num_days=30, data_mode="aws_cur")
    assert aws_fc["data_mode"] == "aws_cur"
    assert "chronological_80_20_metrics" in aws_fc

    # Test detect_anomalies_and_waste with gcp_billing
    gcp_anom = detect_anomalies_and_waste(data_mode="gcp_billing")
    assert gcp_anom["data_mode"] == "gcp_billing"

    # Test optimize_workload_schedules with aws_cur
    aws_opt = optimize_workload_schedules(data_mode="aws_cur")
    assert aws_opt["data_mode"] == "aws_cur"

    print("PASS: All 4 MCP tools executed successfully across 'aws_cur' and 'gcp_billing' data modes.")


if __name__ == "__main__":
    test_aws_cur_ingestion_and_carbon()
    test_gcp_billing_ingestion_and_mapping()
    test_schema_validation_and_currency_check()
    test_mcp_tools_with_public_data_modes()
    print("\nALL VERSION 3 DATASET LOADER & MCP TESTS PASSED SUCCESSFULLY! [SUCCESS]")
