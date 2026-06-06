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
 *   BRIDGE_BIND_HOST    (optional, default 127.0.0.1) — bind interface. Prod
 *                       (Box2) sets 0.0.0.0 so cross-host CRM reaches it over TS.
 *   BRIDGE_PORT         (optional, default OS-assigned) — fixed listen port.
 *                       Prod sets a fixed port so CRM can pin WA_BRIDGE_PORT.
 *
 * Port: defaults to OS-assigned (listen(0)); actual port written to
 * <AUTH_ROOT>/bridge.port for downstream discovery either way.
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
const SINGLE_ADMIN_ID = process.env.SINGLE_ADMIN_ID?.trim() || undefined;

if (!SERVICE_SECRET) {
  console.error('MCP_SERVICE_TOKEN is required. Set the same value used by the CRM MCP server.');
  process.exit(1);
}

// Bind host: default loopback (dev). When set, must be non-empty — a blank value
// is a config mistake, not a request for the default, so fail fast rather than
// silently bind loopback when prod meant 0.0.0.0.
const rawBindHost = process.env.BRIDGE_BIND_HOST;
if (rawBindHost !== undefined && rawBindHost.trim() === '') {
  console.error('BRIDGE_BIND_HOST is set but empty. Unset it to use the default 127.0.0.1, or provide a valid host/IP to bind (prod uses 0.0.0.0).');
  process.exit(1);
}
const BIND_HOST = rawBindHost?.trim() || '127.0.0.1';

// Port: default OS-assigned (dev, via portfile). When set, must be a valid TCP
// port — never silently fall back to a random port when prod pinned a fixed one.
const rawPort = process.env.BRIDGE_PORT?.trim();
let BRIDGE_PORT = 0;
if (rawPort) {
  const parsed = Number(rawPort);
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > 65535) {
    console.error(`BRIDGE_PORT must be an integer in 1-65535, got: "${rawPort}". Unset it to let the OS pick a free port (dev), or set a fixed port (prod).`);
    process.exit(1);
  }
  BRIDGE_PORT = parsed;
}

if (SINGLE_ADMIN_ID && !/^[a-f0-9]{24}$/.test(SINGLE_ADMIN_ID)) {
  console.error(`SINGLE_ADMIN_ID must be 24-char ObjectId hex, got: ${SINGLE_ADMIN_ID}`);
  process.exit(1);
}

if (SINGLE_ADMIN_ID) {
  console.log(`🐈 nanobot WhatsApp Bridge (single-admin mode: ${SINGLE_ADMIN_ID})`);
} else {
  console.log('🐈 nanobot WhatsApp Bridge (multi-tenant)');
}
console.log('=========================================\n');

const server = new BridgeServer(AUTH_ROOT, SERVICE_SECRET, SINGLE_ADMIN_ID, BIND_HOST, BRIDGE_PORT);

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
