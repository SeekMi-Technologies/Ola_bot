/**
 * WebSocket + HTTP server for the multi-tenant Python ↔ Node bridge.
 *
 * Security:
 * - Binds to BRIDGE_BIND_HOST (default 127.0.0.1). Prod (Box2) sets 0.0.0.0 so
 *   the cross-host CRM (Box1) can reach it over the Tailscale NIC.
 * - Rejects browser-originated WebSocket connections (Origin header present).
 * - Per-admin token = HMAC-SHA256(MCP_SERVICE_TOKEN, adminId) — same secret on
 *   bridge/nanobot/CRM, so all three derive identical tokens with no shared table.
 *
 * Routing (Authorization: Bearer <hmac> for REST; token=<hmac> query for WS):
 * - WS:     /wa/<adminId>?token=<hmac>  — per-admin bidirectional stream
 * - POST    /wa/<adminId>/login         — CRM triggers connect; QR arrives via snapshot
 * - GET     /wa/<adminId>/status        — { status, qr? } from snapshot
 * - DELETE  /wa/<adminId>               — disconnect + wipe authDir (logout)
 *
 * Lifecycle:
 * - WhatsAppClient is lazily created on first WS attach OR POST /login for an adminId.
 * - Per-admin subscriber Set<WebSocket>; broadcasts go only to that adminId's set.
 * - Per-client init wrapped in try/catch so one admin's failure can't cascade.
 */

import { createServer as createHttpServer, IncomingMessage, ServerResponse, Server as HttpServer } from 'http';
import { mkdirSync, writeFileSync, rmSync } from 'fs';
import { WebSocketServer, WebSocket } from 'ws';
import { createHmac, timingSafeEqual } from 'crypto';
import { join, dirname } from 'path';
import type { Socket } from 'net';
import { WhatsAppClient, InboundMessage } from './whatsapp.js';

interface SendCommand { type: 'send'; to: string; text: string; }
interface SendMediaCommand {
  type: 'send_media';
  to: string;
  filePath: string;
  mimetype: string;
  caption?: string;
  fileName?: string;
}
type BridgeCommand = SendCommand | SendMediaCommand;

interface BridgeMessage {
  type: 'message' | 'status' | 'qr' | 'error' | 'sent';
  [key: string]: unknown;
}

// adminId is a 24-char ObjectId hex; this is a format guard, not a router.
const ADMIN_ID_RE = /^[a-f0-9]{24}$/;

export interface WaPath {
  adminId: string;
  action: 'login' | 'status' | null; // null = bare /wa/<id> (WS upgrade or DELETE)
  queryToken: string | null;
}

const VALID_ACTIONS = new Set(['login', 'status']);

/**
 * Parse /wa/<adminId>[/<action>][?token=...] via the URL API.
 * Method-based dispatch lives in the caller — this only extracts the shape.
 * Unknown action segments are rejected (return null → 404) so we don't leak
 * that the path structure was valid.
 */
export function parseWaPath(rawUrl: string | undefined): WaPath | null {
  if (!rawUrl) return null;
  let u: URL;
  try {
    u = new URL(rawUrl, 'http://127.0.0.1');
  } catch {
    return null;
  }
  const segs = u.pathname.split('/').filter(Boolean);
  if (segs.length < 2 || segs.length > 3 || segs[0] !== 'wa') return null;
  if (!ADMIN_ID_RE.test(segs[1])) return null;
  const seg = segs[2];
  if (seg !== undefined && !VALID_ACTIONS.has(seg)) return null;
  const action = (seg ?? null) as 'login' | 'status' | null;
  return { adminId: segs[1], action, queryToken: u.searchParams.get('token') };
}

// Per-admin live snapshot, served over REST. The bridge is the source of truth;
// CRM/Integration mirrors this on pull.
//
// No phone number: WhatsApp now exposes a privacy LID (not the real MSISDN) on
// sock.user.id, so we don't surface an identifier we can't trust (see H8 / Baileys #2263).
interface WaSnapshot {
  status: string;  // disconnected | qr_pending | connected | logged_out
  qr?: string;     // present while qr_pending
}

export class BridgeServer {
  private clients: Map<string, WhatsAppClient> = new Map();
  private sockets: Map<string, Set<WebSocket>> = new Map();
  private state: Map<string, WaSnapshot> = new Map();
  private http: HttpServer | null = null;
  private wss: WebSocketServer | null = null;

  /**
   * @param singleAdminId when set, this bridge only serves that adminId — rejects
   *   ws upgrades for any other adminId, and writes portfile to a per-admin path
   *   (`<authRoot>/<adminId>/port`) so nanobot can route each admin to its own
   *   bridge process. Use this for "one terminal = one admin" multi-bridge mode.
   *   When undefined, runs in shared multi-tenant mode (one bridge serves N admins,
   *   portfile at `<authRoot>/bridge.port`).
   */
  constructor(
    private authRoot: string,
    private serviceSecret: string,
    private singleAdminId?: string,
    // Bind interface + fixed port. Defaults preserve dev behavior (loopback +
    // OS-assigned port discovered via portfile). index.ts validates env and
    // passes prod values; callers that omit these (e.g. tests) get the defaults.
    private bindHost: string = '127.0.0.1',
    private port: number = 0,
  ) {}

  /** Per-admin token: HMAC-SHA256(MCP_SERVICE_TOKEN, adminId). */
  private tokenFor(adminId: string): string {
    return createHmac('sha256', this.serviceSecret).update(adminId).digest('hex');
  }

  /** Constant-time HMAC token compare. Both args are 64-char lowercase hex. */
  private tokensMatch(expected: string, actual: string): boolean {
    if (expected.length !== actual.length) return false;
    return timingSafeEqual(Buffer.from(expected, 'hex'), Buffer.from(actual, 'hex'));
  }

  private portFilePath(): string {
    return this.singleAdminId
      ? join(this.authRoot, this.singleAdminId, 'port')
      : join(this.authRoot, 'bridge.port');
  }

  async start(): Promise<void> {
    if (!this.serviceSecret.trim()) {
      throw new Error('MCP_SERVICE_TOKEN is required');
    }

    this.http = createHttpServer((req, res) => this.handleRest(req, res));
    this.wss = new WebSocketServer({ noServer: true });

    this.http.on('upgrade', (req, socket, head) => this.handleUpgrade(req, socket as Socket, head));

    // port=0 → OS picks any free port (dev); fixed port → prod. Either way the
    // actual bound port is written to the portfile for same-host discovery.
    await new Promise<void>((resolve, reject) => {
      this.http!.once('error', reject);
      this.http!.listen(this.port, this.bindHost, () => {
        const addr = this.http!.address();
        if (addr && typeof addr === 'object') {
          const port = addr.port;
          const portFile = this.portFilePath();
          mkdirSync(dirname(portFile), { recursive: true });
          writeFileSync(portFile, String(port), { encoding: 'utf-8' });
          console.log(`🌉 Bridge listening on ws://${this.bindHost}:${port}`);
          console.log(`📂 authRoot=${this.authRoot}`);
          console.log(`📄 portfile=${portFile}`);
          if (this.singleAdminId) {
            console.log(`🔒 Restricted to adminId=${this.singleAdminId}`);
          }
          resolve();
        } else {
          reject(new Error('Failed to determine bound port'));
        }
      });
    });
  }

  private handleUpgrade(req: IncomingMessage, socket: Socket, head: Buffer): void {
    // Reject browser-originated connections
    const origin = req.headers.origin || (req.headers as any).Origin;
    if (origin) {
      console.warn(`Rejected WS upgrade with Origin: ${origin}`);
      socket.destroy();
      return;
    }
    const p = parseWaPath(req.url);
    // WS only on bare /wa/<id> with a token; /login & /status are REST.
    if (!p || p.action !== null || !p.queryToken) {
      socket.destroy();
      return;
    }
    // Single-admin mode: reject any other adminId
    if (this.singleAdminId && p.adminId !== this.singleAdminId) {
      console.warn(`Rejected WS upgrade for ${p.adminId}: bridge restricted to ${this.singleAdminId}`);
      socket.destroy();
      return;
    }
    if (!this.tokensMatch(this.tokenFor(p.adminId), p.queryToken)) {
      console.warn(`Rejected WS upgrade for ${p.adminId}: bad token`);
      socket.destroy();
      return;
    }
    this.wss!.handleUpgrade(req, socket, head, (ws) => this.attachWs(p.adminId, ws));
  }

  private attachWs(adminId: string, ws: WebSocket): void {
    let set = this.sockets.get(adminId);
    if (!set) {
      set = new Set();
      this.sockets.set(adminId, set);
    }
    set.add(ws);
    console.log(`🔗 [${adminId}] Python client attached (${set.size} subscriber${set.size > 1 ? 's' : ''})`);

    // Lazily create Baileys client; safe to call repeatedly
    this.getOrCreateClient(adminId);

    ws.on('message', async (data) => {
      try {
        const cmd = JSON.parse(data.toString()) as BridgeCommand;
        await this.handleCommand(adminId, cmd);
        ws.send(JSON.stringify({ type: 'sent', to: (cmd as any).to } as BridgeMessage));
      } catch (err) {
        console.error(`[${adminId}] command error:`, err);
        ws.send(JSON.stringify({ type: 'error', error: String(err) } as BridgeMessage));
      }
    });

    ws.on('close', () => {
      set!.delete(ws);
      console.log(`🔌 [${adminId}] Python client detached (${set!.size} left)`);
    });

    ws.on('error', (err) => {
      console.error(`[${adminId}] ws error:`, err);
      set!.delete(ws);
    });
  }

  private getOrCreateClient(adminId: string): WhatsAppClient {
    const existing = this.clients.get(adminId);
    if (existing) return existing;

    const dataDir = join(this.authRoot, adminId);
    const c = new WhatsAppClient({
      adminId,
      dataDir,
      onMessage: (msg: InboundMessage) => this.broadcastTo(adminId, { type: 'message', ...msg }),
      onQR: (qr: string) => {
        this.setState(adminId, { status: 'qr_pending', qr });
        this.broadcastTo(adminId, { type: 'qr', qr });
      },
      onStatus: (status: string) => {
        this.broadcastTo(adminId, { type: 'status', status });
        if (status === 'logged_out') {
          // Terminal (401): session is dead. Drop the client + creds so the next
          // login starts fresh and emits a new QR (instead of reusing a dead client).
          this.cleanupAdmin(adminId);
          this.setState(adminId, { status: 'logged_out' });
        } else if (status === 'connected') {
          // Clear the QR once connected; it is stale and shouldn't be served.
          this.setState(adminId, { status, qr: undefined });
        } else {
          this.setState(adminId, { status });
        }
      },
    });
    this.clients.set(adminId, c);
    // Set the initial snapshot synchronously so restLogin reflects a real
    // qr_pending immediately — onQR fires async, after connect() returns.
    this.setState(adminId, { status: 'qr_pending' });

    // Per-client error isolation: a single admin's failure must not cascade.
    c.connect().catch((err) => {
      console.error(`[${adminId}] Baileys init failed:`, err);
      this.setState(adminId, { status: 'disconnected' });
      this.broadcastTo(adminId, { type: 'status', status: 'disconnected', error: String(err) });
    });
    return c;
  }

  private async handleCommand(adminId: string, cmd: BridgeCommand): Promise<void> {
    const client = this.clients.get(adminId);
    if (!client) throw new Error(`No Baileys client for admin ${adminId}`);
    if (cmd.type === 'send') {
      await client.sendMessage(cmd.to, cmd.text);
    } else if (cmd.type === 'send_media') {
      await client.sendMedia(cmd.to, cmd.filePath, cmd.mimetype, cmd.caption, cmd.fileName);
    } else {
      throw new Error(`Unknown command type: ${(cmd as any).type}`);
    }
  }

  private broadcastTo(adminId: string, msg: BridgeMessage): void {
    const set = this.sockets.get(adminId);
    if (!set || set.size === 0) return;
    const data = JSON.stringify(msg);
    for (const ws of set) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(data);
      }
    }
  }

  private setState(adminId: string, patch: Partial<WaSnapshot>): void {
    const prev = this.state.get(adminId) ?? { status: 'disconnected' };
    const next: WaSnapshot = { ...prev, ...patch };
    // A patched `qr: undefined` should remove the key, not store an explicit undefined.
    if ('qr' in patch && patch.qr === undefined) delete next.qr;
    this.state.set(adminId, next);
  }

  private handleRest(req: IncomingMessage, res: ServerResponse): void {
    const p = parseWaPath(req.url);
    if (!p) {
      res.writeHead(404).end();
      return;
    }
    if (this.singleAdminId && p.adminId !== this.singleAdminId) {
      res.writeHead(404).end();
      return;
    }
    const m = req.headers.authorization?.match(/^Bearer ([a-f0-9]{64})$/);
    if (!m || !this.tokensMatch(this.tokenFor(p.adminId), m[1])) {
      res.writeHead(401).end();
      return;
    }

    if (req.method === 'POST' && p.action === 'login') return this.restLogin(p.adminId, res);
    if (req.method === 'GET' && p.action === 'status') return this.restStatus(p.adminId, res);
    if (req.method === 'DELETE' && p.action === null) {
      void this.restLogout(p.adminId, res);
      return;
    }
    res.writeHead(405).end();
  }

  /** POST /login — lazily create + connect the client; QR lands in the snapshot via onQR. */
  private restLogin(adminId: string, res: ServerResponse): void {
    this.getOrCreateClient(adminId);
    const snap = this.state.get(adminId);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: snap?.status ?? 'qr_pending' }));
  }

  /** GET /status — return the live snapshot (qr only while pending). */
  private restStatus(adminId: string, res: ServerResponse): void {
    const snap = this.state.get(adminId) ?? { status: 'disconnected' };
    const body: WaSnapshot = { status: snap.status };
    if (snap.status === 'qr_pending' && snap.qr) body.qr = snap.qr;
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(body));
  }

  /** Drop everything for an admin: WS subscribers, the Baileys client, on-disk auth. */
  private cleanupAdmin(adminId: string): void {
    const subs = this.sockets.get(adminId);
    if (subs) {
      for (const ws of subs) ws.close();
      this.sockets.delete(adminId);
    }
    this.clients.delete(adminId);
    // Remove on-disk auth so nanobot's fs-scan drops the channel and next login re-scans.
    rmSync(join(this.authRoot, adminId), { recursive: true, force: true });
  }

  /** DELETE /wa/<id> — disconnect, drop subscribers, wipe authDir so fs-scan unloads it. */
  private async restLogout(adminId: string, res: ServerResponse): Promise<void> {
    const client = this.clients.get(adminId);
    if (client) {
      try {
        await client.disconnect();
      } catch (err) {
        console.error(`[${adminId}] disconnect during logout failed:`, err);
      }
    }
    this.cleanupAdmin(adminId);
    this.state.set(adminId, { status: 'logged_out' });
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'logged_out' }));
  }

  async stop(): Promise<void> {
    // Close all WS subscribers
    for (const [adminId, set] of this.sockets) {
      for (const ws of set) ws.close();
      console.log(`Closed subscribers for ${adminId}`);
    }
    this.sockets.clear();

    if (this.wss) {
      this.wss.close();
      this.wss = null;
    }

    // Disconnect all Baileys clients
    for (const [adminId, client] of this.clients) {
      try {
        await client.disconnect();
        console.log(`Disconnected ${adminId}`);
      } catch (err) {
        console.error(`Error disconnecting ${adminId}:`, err);
      }
    }
    this.clients.clear();

    if (this.http) {
      await new Promise<void>((resolve) => this.http!.close(() => resolve()));
      this.http = null;
    }
  }
}
