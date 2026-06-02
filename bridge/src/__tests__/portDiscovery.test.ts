/**
 * Unit tests for portDiscovery and the bridge's listen(0) strategy.
 *
 * Why these tests:
 * - OS port auto-assignment (listen(0)) is core to the multi-tenant bridge —
 *   we don't want to hardcode a port range that might collide with the user's
 *   other dev servers (Grafana, react-dev, etc.).
 * - These tests confirm the OS really hands out distinct free ports and that
 *   we can round-trip them through the portfile so nanobot/CRM can discover.
 *
 * Runner: built-in `node:test` (Node >= 18). Run via `npm test`.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { createServer } from 'http';
import type { AddressInfo } from 'net';
import { writePortFile, readPortFile } from '../portDiscovery.js';

function withTmpDir(fn: (dir: string) => void | Promise<void>): () => Promise<void> {
  return async () => {
    const dir = mkdtempSync(join(tmpdir(), 'bridge-test-'));
    try {
      await fn(dir);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  };
}

test('writePortFile then readPortFile round-trips', withTmpDir((dir) => {
  writePortFile(dir, 12345);
  assert.equal(readPortFile(dir), 12345);
}));

test('writePortFile creates authRoot if missing', withTmpDir((dir) => {
  const nested = join(dir, 'deep', 'nested', 'dir');
  writePortFile(nested, 54321);
  assert.equal(readPortFile(nested), 54321);
}));

test('readPortFile returns null when file missing', withTmpDir((dir) => {
  assert.equal(readPortFile(dir), null);
}));

test('readPortFile returns null on non-numeric content', withTmpDir((dir) => {
  writeFileSync(join(dir, 'bridge.port'), 'not-a-number');
  assert.equal(readPortFile(dir), null);
}));

test('readPortFile returns null on zero or negative port', withTmpDir((dir) => {
  writeFileSync(join(dir, 'bridge.port'), '0');
  assert.equal(readPortFile(dir), null);
  writeFileSync(join(dir, 'bridge.port'), '-1');
  assert.equal(readPortFile(dir), null);
}));

test('readPortFile trims whitespace/newlines', withTmpDir((dir) => {
  writeFileSync(join(dir, 'bridge.port'), '  9876\n');
  assert.equal(readPortFile(dir), 9876);
}));

test('listen(0) binds to non-zero ephemeral port and accepts connections', async () => {
  const server = createServer((_req, res) => res.end('ok'));
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolve());
  });
  const addr = server.address() as AddressInfo;
  assert.ok(addr && typeof addr === 'object', 'expected AddressInfo object');
  const port = addr.port;
  assert.ok(port > 0, `port should be > 0, got ${port}`);
  assert.ok(port >= 1024, `OS-assigned ephemeral ports are typically >= 1024, got ${port}`);

  // Verify client can actually connect end-to-end
  const r = await fetch(`http://127.0.0.1:${port}`);
  assert.equal(r.status, 200);
  assert.equal(await r.text(), 'ok');

  await new Promise<void>((resolve) => server.close(() => resolve()));
});

test('two simultaneous listen(0) calls get distinct ports — no conflict', async () => {
  const s1 = createServer((_req, res) => res.end('s1'));
  const s2 = createServer((_req, res) => res.end('s2'));
  await new Promise<void>((resolve) => s1.listen(0, '127.0.0.1', () => resolve()));
  await new Promise<void>((resolve) => s2.listen(0, '127.0.0.1', () => resolve()));
  const p1 = (s1.address() as AddressInfo).port;
  const p2 = (s2.address() as AddressInfo).port;
  assert.notEqual(p1, p2, `expected different ports, got both ${p1}`);

  // Both servers accept connections independently
  const [r1, r2] = await Promise.all([
    fetch(`http://127.0.0.1:${p1}`).then((r) => r.text()),
    fetch(`http://127.0.0.1:${p2}`).then((r) => r.text()),
  ]);
  assert.equal(r1, 's1');
  assert.equal(r2, 's2');

  await Promise.all([
    new Promise<void>((resolve) => s1.close(() => resolve())),
    new Promise<void>((resolve) => s2.close(() => resolve())),
  ]);
});

test('listen(0) on already-bound port range — OS picks next free', async () => {
  // Bind 10 servers in a row; all should get distinct ports without our intervention.
  const servers = Array.from({ length: 10 }, () => createServer((_req, res) => res.end()));
  await Promise.all(servers.map((s) => new Promise<void>((resolve) => s.listen(0, '127.0.0.1', () => resolve()))));
  const ports = servers.map((s) => (s.address() as AddressInfo).port);
  const unique = new Set(ports);
  assert.equal(unique.size, 10, `expected 10 distinct ports, got ${ports.join(',')}`);
  await Promise.all(servers.map((s) => new Promise<void>((resolve) => s.close(() => resolve()))));
});

test('end-to-end: bridge-style flow — listen(0) + writePortFile + readPortFile finds it', withTmpDir(async (dir) => {
  const server = createServer((_req, res) => res.end('bridge-ok'));
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', () => resolve()));
  const port = (server.address() as AddressInfo).port;

  // Bridge writes; consumer (nanobot/CRM) reads
  writePortFile(dir, port);
  const discovered = readPortFile(dir);
  assert.equal(discovered, port);

  // Consumer hits the discovered port → reaches our server
  const r = await fetch(`http://127.0.0.1:${discovered}`);
  assert.equal(await r.text(), 'bridge-ok');

  await new Promise<void>((resolve) => server.close(() => resolve()));
}));
