/**
 * Unit tests for bridge REST routing + auth (B0 / Ola #326).
 *
 * Two layers:
 *  1. parseWaPath — the URL-based router that replaced the old tri-purpose
 *     regex. Pure function, exhaustively tested.
 *  2. BridgeServer REST — start a real server in a tmp authRoot and hit it over
 *     HTTP. Covers auth (401), status snapshot, logout, and method/path misses
 *     (404/405). The happy-path POST /login spins a real Baileys socket (network),
 *     so it is verified by manual curl E2E, not here — but login's 401 auth gate
 *     is covered (it rejects before any client is created).
 *
 * Runner: built-in `node:test`. Run via `npm test`.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, readFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { createHmac } from 'crypto';
import { createServer as createNetServer } from 'net';
import { BridgeServer, parseWaPath } from '../server.js';

const ID = 'a'.repeat(24); // valid 24-hex adminId
const HMAC64 = 'b'.repeat(64); // a syntactically valid 64-hex token

// ---- parseWaPath ----------------------------------------------------------

test('parseWaPath: bare /wa/<id>?token= → WS shape (action null, token set)', () => {
  const p = parseWaPath(`/wa/${ID}?token=${HMAC64}`);
  assert.deepEqual(p, { adminId: ID, action: null, queryToken: HMAC64 });
});

test('parseWaPath: /wa/<id>/login → action login', () => {
  const p = parseWaPath(`/wa/${ID}/login`);
  assert.deepEqual(p, { adminId: ID, action: 'login', queryToken: null });
});

test('parseWaPath: /wa/<id>/status → action status', () => {
  const p = parseWaPath(`/wa/${ID}/status`);
  assert.deepEqual(p, { adminId: ID, action: 'status', queryToken: null });
});

test('parseWaPath: bare /wa/<id> (no query) → action null, token null (DELETE)', () => {
  const p = parseWaPath(`/wa/${ID}`);
  assert.deepEqual(p, { adminId: ID, action: null, queryToken: null });
});

test('parseWaPath: rejects non-24-hex adminId', () => {
  assert.equal(parseWaPath('/wa/not-an-id/status'), null);
  assert.equal(parseWaPath(`/wa/${'a'.repeat(23)}/status`), null);
  assert.equal(parseWaPath(`/wa/${'A'.repeat(24)}`), null); // uppercase not allowed
});

test('parseWaPath: rejects extra segments and wrong prefix', () => {
  assert.equal(parseWaPath(`/wa/${ID}/status/extra`), null);
  assert.equal(parseWaPath(`/foo/${ID}/status`), null);
  assert.equal(parseWaPath(`/wa`), null);
});

test('parseWaPath: undefined / garbage → null', () => {
  assert.equal(parseWaPath(undefined), null);
  assert.equal(parseWaPath(''), null);
});

test('parseWaPath: rejects unknown action segment', () => {
  assert.equal(parseWaPath(`/wa/${ID}/hacked`), null);
  assert.equal(parseWaPath(`/wa/${ID}/logout`), null); // DELETE is bare /wa/<id>, not an action
});

// ---- BridgeServer REST ----------------------------------------------------

const SECRET = 'test-service-secret-0123456789ab';

function withServer(
  fn: (ctx: {
    base: string;
    tokenFor: (id: string) => string;
    authRoot: string;
  }) => Promise<void>
): () => Promise<void> {
  return async () => {
    const authRoot = mkdtempSync(join(tmpdir(), 'bridge-srv-'));
    const server = new BridgeServer(authRoot, SECRET);
    const tokenFor = (id: string) => createHmac('sha256', SECRET).update(id).digest('hex');
    try {
      // start() inside try so a bind failure still hits the finally (no tempdir leak).
      await server.start();
      const port = Number(readFileSync(join(authRoot, 'bridge.port'), 'utf-8').trim());
      await fn({ base: `http://127.0.0.1:${port}`, tokenFor, authRoot });
    } finally {
      await server.stop().catch((e) => console.warn('test server stop failed:', (e as Error)?.message));
      rmSync(authRoot, { recursive: true, force: true });
    }
  };
}

// ---- fixed bind/port (2a) -------------------------------------------------

/** Probe a free ephemeral port by binding :0, reading it, then releasing it. */
function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const probe = createNetServer();
    probe.once('error', reject);
    probe.listen(0, '127.0.0.1', () => {
      const addr = probe.address();
      if (addr && typeof addr === 'object') probe.close(() => resolve(addr.port));
      else reject(new Error('could not determine free port'));
    });
  });
}

test('BridgeServer honors a fixed port + bindHost (portfile records the configured port)', async () => {
  const authRoot = mkdtempSync(join(tmpdir(), 'bridge-port-'));
  const port = await freePort();
  const server = new BridgeServer(authRoot, SECRET, undefined, '127.0.0.1', port);
  try {
    await server.start();
    const written = Number(readFileSync(join(authRoot, 'bridge.port'), 'utf-8').trim());
    assert.equal(written, port, 'portfile must record the configured fixed port, not a random one');
    const res = await fetch(`http://127.0.0.1:${port}/wa/${ID}/status`, {
      headers: { Authorization: `Bearer ${createHmac('sha256', SECRET).update(ID).digest('hex')}` },
    });
    assert.equal(res.status, 200, 'server must actually be listening on the configured port');
  } finally {
    await server.stop().catch((e) => console.warn('test server stop failed:', (e as Error)?.message));
    rmSync(authRoot, { recursive: true, force: true });
  }
});

test(
  'REST GET /status with valid token, no client → 200 {status:disconnected}',
  withServer(async ({ base, tokenFor }) => {
    const res = await fetch(`${base}/wa/${ID}/status`, {
      headers: { Authorization: `Bearer ${tokenFor(ID)}` },
    });
    assert.equal(res.status, 200);
    assert.deepEqual(await res.json(), { status: 'disconnected' });
  })
);

test(
  'REST GET /status with wrong token → 401',
  withServer(async ({ base }) => {
    const res = await fetch(`${base}/wa/${ID}/status`, {
      headers: { Authorization: `Bearer ${'c'.repeat(64)}` },
    });
    assert.equal(res.status, 401);
  })
);

test(
  'REST GET /status with no Authorization header → 401',
  withServer(async ({ base }) => {
    const res = await fetch(`${base}/wa/${ID}/status`);
    assert.equal(res.status, 401);
  })
);

test(
  'REST POST /login with wrong token → 401 (auth gate before client creation)',
  withServer(async ({ base }) => {
    const res = await fetch(`${base}/wa/${ID}/login`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${'c'.repeat(64)}` },
    });
    assert.equal(res.status, 401);
  })
);

test(
  'REST DELETE /wa/<id> with valid token → 200 {status:logged_out}',
  withServer(async ({ base, tokenFor }) => {
    const res = await fetch(`${base}/wa/${ID}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${tokenFor(ID)}` },
    });
    assert.equal(res.status, 200);
    assert.deepEqual(await res.json(), { status: 'logged_out' });
  })
);

test(
  'REST unknown method on valid path → 405',
  withServer(async ({ base, tokenFor }) => {
    const res = await fetch(`${base}/wa/${ID}/status`, {
      method: 'PUT',
      headers: { Authorization: `Bearer ${tokenFor(ID)}` },
    });
    assert.equal(res.status, 405);
  })
);

test(
  'REST malformed path (bad adminId) → 404',
  withServer(async ({ base, tokenFor }) => {
    const res = await fetch(`${base}/wa/not-an-id/status`, {
      headers: { Authorization: `Bearer ${tokenFor(ID)}` },
    });
    assert.equal(res.status, 404);
  })
);
