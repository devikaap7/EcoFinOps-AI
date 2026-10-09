"""
MCP Stdio Transport Client Test Script.
Launches mcp_server.py in a subprocess over stdio transport, performs JSON-RPC
initialization & tool discovery (tools/list), and invokes a tool (tools/call).
"""

import sys
import os
import json
import subprocess

print("=== TESTING MCP STDIO TRANSPORT CLIENT HANDSHAKE & DISCOVERY ===")

python_exe = sys.executable
server_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mcp_server.py")
env = os.environ.copy()
env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

process = subprocess.Popen(
    [python_exe, server_script],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env=env,
    bufsize=1,
)

def send_request(req):
    body = json.dumps(req)
    process.stdin.write(f"{body}\n")
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        err = process.stderr.read()
        raise RuntimeError(f"Server closed connection. Stderr: {err}")
    return json.loads(line)

# Step 1: Initialize Request
init_req = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test-mcp-client", "version": "1.0.0"}
    }
}

init_res = send_request(init_req)
print("\n--- Step 1: JSON-RPC Initialize Response ---")
print("Protocol Version:", init_res.get("result", {}).get("protocolVersion"))
print("Server Info:", init_res.get("result", {}).get("serverInfo"))

# Send initialized notification
initialized_notif = {
    "jsonrpc": "2.0",
    "method": "notifications/initialized"
}
process.stdin.write(f"{json.dumps(initialized_notif)}\n")
process.stdin.flush()

# Step 2: Tools Discovery (tools/list)
list_tools_req = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list"
}

list_res = send_request(list_tools_req)
tools = list_res.get("result", {}).get("tools", [])
tool_names = [t["name"] for t in tools]
print("\n--- Step 2: Discovered MCP Tools (tools/list) ---")
print("Discovered Tools Count:", len(tools))
print("Tool Names:", tool_names)

assert len(tools) == 5, f"Expected 5 tools, discovered {len(tools)}"
assert "get_telemetry_summary" in tool_names
assert "forecast_cloud_spend" in tool_names
assert "detect_anomalies_and_waste" in tool_names
assert "optimize_workload_schedules" in tool_names
assert "get_carbon_methodology" in tool_names

# Step 3: Invoke Tool (tools/call)
call_tool_req = {
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
        "name": "get_telemetry_summary",
        "arguments": {"num_days": 7, "seed": 42}
    }
}

call_res = send_request(call_tool_req)
print("\n--- Step 3: Invoke get_telemetry_summary via JSON-RPC tools/call ---")
content = call_res.get("result", {}).get("content", [])
print("Invocation Result Snippet:", content[0]["text"][:200] if content else "Empty content")

# Terminate process
process.terminate()
print("\n=== MCP STDIO PROTOCOL CLIENT VERIFICATION SUCCESSFUL! ===")
