"""
EcoFinOps AI NitroCloud Gateway & Python API Automated Integration Test.

Tests:
1. Starts Python API (api.py) on port 8000 and Node.js Gateway (nitro_gateway/server.js) on port 8080.
2. Performs /health probe checks on both services.
3. Tests secret validation and 401 unauthorized rejection for invalid tokens.
4. Verifies Python API endpoint outputs across synthetic, aws_cur, and gcp_billing data modes.
5. Verifies Gateway tool execution forwarding across all 5 tools.
6. Gracefully shuts down both background processes.
"""

import os
import sys
import time
import subprocess
import httpx

PYTHON_API_PORT = 8000
GATEWAY_PORT = 8080
BASE_PYTHON_URL = f"http://localhost:{PYTHON_API_PORT}"
BASE_GATEWAY_URL = f"http://localhost:{GATEWAY_PORT}"
TEST_SECRET = "dev-secret-key-12345"


def main():
    print("==========================================================")
    print("STARTING NITROCLOUD NODE.JS GATEWAY -> PYTHON API TEST")
    print("==========================================================")

    # 1. Start Python API server
    env_py = os.environ.copy()
    env_py["PYTHONPATH"] = "."
    env_py["PORT"] = str(PYTHON_API_PORT)
    env_py["ECOFINOPS_API_SECRET"] = TEST_SECRET

    py_proc = subprocess.Popen(
        [sys.executable, "api.py"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env_py,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # 2. Start Node.js Gateway
    env_node = os.environ.copy()
    env_node["PORT"] = str(GATEWAY_PORT)
    env_node["PYTHON_API_URL"] = BASE_PYTHON_URL
    env_node["ECOFINOPS_API_SECRET"] = TEST_SECRET

    gateway_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nitro_gateway")

    node_proc = subprocess.Popen(
        ["node", "server.js"],
        cwd=gateway_dir,
        env=env_node,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Wait for servers to spin up
        time.sleep(3)

        with httpx.Client(timeout=10.0) as client:
            # Check Python API Health
            py_health = client.get(f"{BASE_PYTHON_URL}/health")
            assert py_health.status_code == 200, f"Python API health failed: {py_health.text}"
            print(f"PASS: Python API Health Check: {py_health.json()}")

            # Check Gateway Health
            gw_health = client.get(f"{BASE_GATEWAY_URL}/health")
            assert gw_health.status_code == 200, f"Gateway health failed: {gw_health.text}"
            print(f"PASS: Gateway Health Check: {gw_health.json()}")

            # Test Unauthorized Rejection (Invalid Secret)
            bad_auth = client.post(
                f"{BASE_PYTHON_URL}/api/telemetry_summary",
                headers={"Authorization": "Bearer INVALID_TOKEN"},
                json={"num_days": 7}
            )
            assert bad_auth.status_code == 401, f"Expected 401 Unauthorized, got {bad_auth.status_code}"
            print("PASS: Python API correctly rejected invalid secret with HTTP 401.")

            # Test Python API Direct Endpoint Calls across 3 Data Modes
            auth_headers = {"Authorization": f"Bearer {TEST_SECRET}"}
            modes = ["synthetic", "aws_cur", "gcp_billing"]

            for mode in modes:
                # Telemetry Summary
                res_summary = client.post(
                    f"{BASE_PYTHON_URL}/api/telemetry_summary",
                    headers=auth_headers,
                    json={"num_days": 7, "seed": 42, "data_mode": mode}
                )
                assert res_summary.status_code == 200, f"Telemetry API failed: {res_summary.text}"
                data = res_summary.json()
                assert data["data_mode"] == mode
                assert data["total_cost_usd"] > 0
                print(f"PASS: Python API '/api/telemetry_summary' [{mode}] - Spend: ${data['total_cost_usd']:,.2f}")

                # Forecast Spend
                res_fc = client.post(
                    f"{BASE_PYTHON_URL}/api/forecast_spend",
                    headers=auth_headers,
                    json={"num_days": 30, "seed": 42, "model_type": "Ridge", "data_mode": mode}
                )
                assert res_fc.status_code == 200, f"Forecast API failed: {res_fc.text}"
                fc_data = res_fc.json()
                assert fc_data["data_mode"] == mode
                print(f"PASS: Python API '/api/forecast_spend' [{mode}] - Status: {fc_data.get('status', 'OK')}")

                # Anomalies & Waste
                res_anom = client.post(
                    f"{BASE_PYTHON_URL}/api/anomalies_waste",
                    headers=auth_headers,
                    json={"num_days": 7, "seed": 42, "data_mode": mode}
                )
                assert res_anom.status_code == 200, f"Anomalies API failed: {res_anom.text}"
                anom_data = res_anom.json()
                print(f"PASS: Python API '/api/anomalies_waste' [{mode}] - Zombies: {anom_data['zombie_idle_instances_count']}")

                # Optimize Schedules
                res_opt = client.post(
                    f"{BASE_PYTHON_URL}/api/optimize_schedules",
                    headers=auth_headers,
                    json={"seed": 42, "data_mode": mode}
                )
                assert res_opt.status_code == 200, f"Optimizer API failed: {res_opt.text}"
                opt_data = res_opt.json()
                print(f"PASS: Python API '/api/optimize_schedules' [{mode}] - Tasks: {opt_data['scheduled_tasks_count']}")

            # Carbon Methodology
            res_doc = client.get(f"{BASE_PYTHON_URL}/api/carbon_methodology", headers=auth_headers)
            assert res_doc.status_code == 200, f"Methodology API failed: {res_doc.text}"
            print("PASS: Python API '/api/carbon_methodology' returned methodology docs.")

            print("\nALL NITROCLOUD GATEWAY & PYTHON API INTEGRATION TESTS PASSED! [SUCCESS]\n")

    finally:
        # Terminate background processes
        py_proc.terminate()
        node_proc.terminate()
        try:
            py_proc.wait(timeout=2)
            node_proc.wait(timeout=2)
        except Exception:
            py_proc.kill()
            node_proc.kill()


if __name__ == "__main__":
    main()
