/**
 * Port discovery helpers for the multi-tenant bridge.
 *
 * The bridge no longer hardcodes a port. It calls server.listen(0) to let the
 * OS pick any free port (1024+), then writes the actual port to a portfile so
 * nanobot WhatsAppChannel + CRM REST proxy can discover it.
 */

import { mkdirSync, writeFileSync, readFileSync, existsSync } from 'fs';
import { join } from 'path';

const PORT_FILENAME = 'bridge.port';

export function writePortFile(authRoot: string, port: number): void {
  mkdirSync(authRoot, { recursive: true });
  writeFileSync(join(authRoot, PORT_FILENAME), String(port), { encoding: 'utf-8' });
}

export function readPortFile(authRoot: string): number | null {
  const p = join(authRoot, PORT_FILENAME);
  if (!existsSync(p)) return null;
  const raw = readFileSync(p, 'utf-8').trim();
  const n = parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : null;
}
