/**
 * WebSocket + HTTP server for the multi-tenant Python ↔ Node bridge.
 *
 * Security:
 * - Binds to 127.0.0.1 only.
 * - Rejects browser-originated WebSocket connections (Origin header present).
 * - Per-admin token = HMAC-SHA256(MCP_SERVICE_TOKEN, adminId) — same secret on
 *   bridge/nanobot/CRM, so all three derive identical tokens with no shared table.
 *
 * Routing:
 * - WS:   /wa/<adminId>?token=<hmac>   — per-admin bidirectional stream
 * - REST: GET /wa/<adminId>/status     — dev debug, Authorization: Bearer <hmac>
 *
 * Lifecycle:
 * - WhatsAppClient is lazily created on first WS attach for an adminId.
 * - Per-admin subscriber Set<WebSocket>; broadcasts go only to that adminId's set.
 * - Per-client init wrapped in try/catch so one admin's failure can't cascade.
 */

import { createServer as createHttpServer, IncomingMessage, ServerResponse, Server as HttpServer } from 'http';
import { mkdirSync, writeFileSync } from 'fs';
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

// 24-char ObjectId hex for adminId; 64-char hex for HMAC-SHA256 token
const PATH_RE = /^\/wa\/([a-f0-9]{24})(?:\/(status))?(?:\?token=([a-f0-9]{64}))?$/;

interface PathParts {
  adminId: string;
  isRest: boolean;  // true if /status suffix present (REST), false for bare /wa/<id> (WS)
  queryToken?: string;
}

function parsePath(url: string | undefined): PathParts | null {
  if (!url) return null;
  const m = url.match(PATH_RE);
  if (!m) return null;
  return { adminId: m[1], isRest: m[2] === 'status', queryToken: m[3] };
}

export class BridgeServer {
  private clients: Map<string, WhatsAppClient> = new Map();
  private sockets: Map<string, Set<WebSocket>> = new Map();
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

    // listen(0) → OS picks any free port; write actual port to portfile
    await new Promise<void>((resolve, reject) => {
      this.http!.once('error', reject);
      this.http!.listen(0, '127.0.0.1', () => {
        const addr = this.http!.address();
        if (addr && typeof addr === 'object') {
          const port = addr.port;
          const portFile = this.portFilePath();
          mkdirSync(dirname(portFile), { recursive: true });
          writeFileSync(portFile, String(port), { encoding: 'utf-8' });
          console.log(`🌉 Bridge listening on ws://127.0.0.1:${port}`);
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
    const p = parsePath(req.url);
    if (!p || p.isRest || !p.queryToken) {
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
      onQR: (qr: string) => this.broadcastTo(adminId, { type: 'qr', qr }),
      onStatus: (status: string) => this.broadcastTo(adminId, { type: 'status', status }),
    });
    this.clients.set(adminId, c);

    // Per-client error isolation: a single admin's failure must not cascade.
    c.connect().catch((err) => {
      console.error(`[${adminId}] Baileys init failed:`, err);
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

  private handleRest(req: IncomingMessage, res: ServerResponse): void {
    // Currently only: GET /wa/<adminId>/status
    if (req.method !== 'GET') {
      res.writeHead(405).end();
      return;
    }
    const p = parsePath(req.url);
    if (!p || !p.isRest) {
      res.writeHead(404).end();
      return;
    }
    const auth = req.headers.authorization;
    const m = auth?.match(/^Bearer ([a-f0-9]{64})$/);
    if (!m || !this.tokensMatch(this.tokenFor(p.adminId), m[1])) {
      res.writeHead(401).end();
      return;
    }
    const client = this.clients.get(p.adminId);
    const subs = this.sockets.get(p.adminId)?.size ?? 0;
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      adminId: p.adminId,
      clientLoaded: !!client,
      subscribers: subs,
    }));
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
