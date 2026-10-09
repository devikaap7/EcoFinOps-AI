/**
 * EcoFinOps AI NitroCloud MCP Streamable HTTP / SSE Gateway.
 * Registers the 5 EcoFinOps MCP tools and forwards tool calls over HTTP
 * to the Python Analytics API with Bearer secret authentication.
 */

const express = require('express');
const { McpServer } = require('@modelcontextprotocol/sdk/server/mcp.js');
const { SSEServerTransport } = require('@modelcontextprotocol/sdk/server/sse.js');
const { z } = require('zod');

const PORT = process.env.PORT || 8080;
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';
const ECOFINOPS_API_SECRET = process.env.ECOFINOPS_API_SECRET || '';

const app = express();
app.use(express.json());

// Public health probe endpoint (for load balancers & NitroCloud health checks)
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'EcoFinOps AI Nitro Gateway',
    python_api_target: PYTHON_API_URL
  });
});

// Helper function to call Python API securely
async function callPythonAPI(endpoint, method = 'POST', body = null) {
  if (!ECOFINOPS_API_SECRET) {
    throw new Error('ECOFINOPS_API_SECRET environment variable is not configured on gateway.');
  }

  const options = {
    method: method,
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${ECOFINOPS_API_SECRET}`
    }
  };

  if (body) {
    options.body = JSON.stringify(body);
  }

  const url = `${PYTHON_API_URL}${endpoint}`;
  const response = await fetch(url, options);

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Python API error (${response.status}): ${errText}`);
  }

  return await response.json();
}

// Initialize McpServer instance
const mcpServer = new McpServer({
  name: 'EcoFinOps AI Gateway',
  version: '3.0.0'
});

// Tool 1: get_telemetry_summary
mcpServer.tool(
  'get_telemetry_summary',
  'Summarizes cloud fleet spend, energy consumption, and carbon footprint.',
  {
    num_days: z.number().default(30).describe('Simulation days (1-90)'),
    seed: z.number().default(42).describe('Random seed for reproducibility'),
    regions: z.array(z.string()).optional().describe('Optional cloud region filter'),
    data_mode: z.enum(['synthetic', 'aws_cur', 'gcp_billing']).default('synthetic').describe('Operating data mode')
  },
  async (args) => {
    try {
      const result = await callPythonAPI('/api/telemetry_summary', 'POST', args);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    } catch (err) {
      return { content: [{ type: 'text', text: `Error calling telemetry summary: ${err.message}` }], isError: true };
    }
  }
);

// Tool 2: forecast_cloud_spend
mcpServer.tool(
  'forecast_cloud_spend',
  'Evaluates out-of-sample time-series cost forecasting comparing ML model to 24h lag baseline.',
  {
    num_days: z.number().default(30).describe('Simulation window duration'),
    seed: z.number().default(42).describe('Random seed'),
    model_type: z.enum(['Ridge', 'RandomForest']).default('Ridge').describe('Forecasting algorithm'),
    train_ratio: z.number().default(0.8).describe('Train ratio (0.5-0.9)'),
    data_mode: z.enum(['synthetic', 'aws_cur', 'gcp_billing']).default('synthetic').describe('Operating data mode')
  },
  async (args) => {
    try {
      const result = await callPythonAPI('/api/forecast_spend', 'POST', args);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    } catch (err) {
      return { content: [{ type: 'text', text: `Error calling forecast spend: ${err.message}` }], isError: true };
    }
  }
);

// Tool 3: detect_anomalies_and_waste
mcpServer.tool(
  'detect_anomalies_and_waste',
  'Detects high-cost anomaly spikes and identifies zombie / underutilized idle instances.',
  {
    num_days: z.number().default(30).describe('Simulation window duration'),
    seed: z.number().default(42).describe('Random seed'),
    contamination: z.number().default(0.03).describe('Anomaly sensitivity fraction'),
    data_mode: z.enum(['synthetic', 'aws_cur', 'gcp_billing']).default('synthetic').describe('Operating data mode')
  },
  async (args) => {
    try {
      const result = await callPythonAPI('/api/anomalies_waste', 'POST', args);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    } catch (err) {
      return { content: [{ type: 'text', text: `Error calling anomaly detection: ${err.message}` }], isError: true };
    }
  }
);

// Tool 4: optimize_workload_schedules
mcpServer.tool(
  'optimize_workload_schedules',
  'Simulates batch workload scheduling optimization comparing Baseline, Cost-Aware, and Carbon-Aware strategies.',
  {
    seed: z.number().default(42).describe('Random seed for workload generation'),
    data_mode: z.enum(['synthetic', 'aws_cur', 'gcp_billing']).default('synthetic').describe('Operating data mode')
  },
  async (args) => {
    try {
      const result = await callPythonAPI('/api/optimize_schedules', 'POST', args);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    } catch (err) {
      return { content: [{ type: 'text', text: `Error calling workload optimizer: ${err.message}` }], isError: true };
    }
  }
);

// Tool 5: get_carbon_methodology
mcpServer.tool(
  'get_carbon_methodology',
  'Returns markdown documentation detailing carbon calculation formulas and PUE assumptions.',
  {},
  async () => {
    try {
      const result = await callPythonAPI('/api/carbon_methodology', 'GET');
      return { content: [{ type: 'text', text: result.methodology_doc || JSON.stringify(result) }] };
    } catch (err) {
      return { content: [{ type: 'text', text: `Error fetching carbon methodology: ${err.message}` }], isError: true };
    }
  }
);

// MCP Transport Sessions
let transports = new Map();

app.get('/sse', async (req, res) => {
  const transport = new SSEServerTransport('/messages', res);
  const sessionId = transport.sessionId;
  transports.set(sessionId, transport);

  req.on('close', () => {
    transports.delete(sessionId);
  });

  await mcpServer.connect(transport);
});

app.post('/messages', async (req, res) => {
  const sessionId = req.query.sessionId;
  const transport = transports.get(sessionId);
  if (!transport) {
    res.status(404).send('Session not found');
    return;
  }
  await transport.handlePostMessage(req, res);
});

app.listen(PORT, () => {
  console.log(`🌿 EcoFinOps AI Nitro Gateway running on port ${PORT}`);
  console.log(`Target Python API: ${PYTHON_API_URL}`);
});
