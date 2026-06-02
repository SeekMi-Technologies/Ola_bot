#!/usr/bin/env node
/**
 * nanobot WhatsApp Bridge — multi-tenant mode
 *
 * Single Node process hosts N WhatsAppClient instances, one per CRM Admin.
 * Routes WebSocket connections by URL path (/wa/<adminId>?token=<hmac>) and
 * uses HMAC-SHA256(MCP_SERVICE_TOKEN, adminId) as the per-admin token so the
 * bridge and nanobot derive identical tokens with no shared table.
 *
 * Env:
 *   MCP_SERVICE_TOKEN   (required) — same secret used by CRM MCP server.
 *                       Derives per-admin tokens via HMAC.
 *   AUTH_ROOT           (optional, default ~/.nanobot/wa) — root dir for
 *                       per-admin auth state. Layout: <root>/<adminId>/{auth,media}.
 *                       Port written to <root>/bridge.port.
 *
 * Port: not hardcoded. OS picks any free port via listen(0); actual port
 * written to <AUTH_ROOT>/bridge.port for downstream discovery.
 */

// Polyfill crypto for Baileys in ESM
import { webcrypto } from 'crypto';
if (!globalThis.crypto) {
  (globalThis as any).crypto = webcrypto;
}

import { BridgeServer } from './server.js';
import { homedir } from 'os';
import { join } from 'path';

const AUTH_ROOT = process.env.AUTH_ROOT || join(homedir(), '.nanobot', 'wa');
const SERVICE_SECRET = process.env.MCP_SERVICE_TOKEN?.trim();

if (!SERVICE_SECRET) {
  console.error('MCP_SERVICE_TOKEN is required. Set the same value used by the CRM MCP server.');
  process.exit(1);
}

console.log('🐈 nanobot WhatsApp Bridge (multi-tenant)');
console.log('=========================================\n');

const server = new BridgeServer(AUTH_ROOT, SERVICE_SECRET);

process.on('SIGINT', async () => {
  console.log('\n\nShutting down...');
  await server.stop();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  await server.stop();
  process.exit(0);
});

server.start().catch((error) => {
  console.error('Failed to start bridge:', error);
  process.exit(1);
});
