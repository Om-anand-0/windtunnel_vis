import type { Settings } from './state';
import { defaultSettings } from './state';

/**
 * Settings persistence (localStorage) and shareable links. Only plain values are stored; uploaded
 * models are not (the link falls back to the default vehicle).
 */
const KEY = 'windtunnel.settings.v1';
const SKIP = new Set(['paused']);

export function loadSaved(): Partial<Settings> {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return {};
    const v = JSON.parse(raw) as Partial<Settings>;
    if (v.vehicle === 'upload') delete v.vehicle;
    return v;
  } catch {
    return {};
  }
}

export function save(s: Settings) {
  try {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(s)) if (!SKIP.has(k)) out[k] = v;
    localStorage.setItem(KEY, JSON.stringify(out));
  } catch {
    /* storage unavailable (private mode) — ignore */
  }
}

export function clearSaved() {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
}

/** Apply a partial settings object, keeping only keys that exist with the same type. */
export function applyPartial(s: Settings, p: Record<string, unknown>) {
  const rec = s as unknown as Record<string, unknown>;
  for (const [k, v] of Object.entries(p)) {
    if (!(k in rec) || SKIP.has(k)) continue;
    if (typeof rec[k] === typeof v) rec[k] = v;
  }
}

/** URL that reproduces the current view: every setting that differs from the defaults + camera. */
export function shareUrl(s: Settings, camera?: { pos: number[]; target: number[] }): string {
  const d = defaultSettings() as unknown as Record<string, unknown>;
  const rec = s as unknown as Record<string, unknown>;
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(rec)) {
    if (SKIP.has(k) || v === d[k]) continue;
    if (k === 'vehicle' && v === 'upload') continue;
    q.set('s.' + k, typeof v === 'boolean' ? (v ? '1' : '0') : typeof v === 'number' ? String(+v.toFixed(4)) : String(v));
  }
  if (camera) q.set('cam', [...camera.pos, ...camera.target].map((x) => x.toFixed(1)).join(','));
  return `${location.origin}${location.pathname}?${q.toString()}`;
}
