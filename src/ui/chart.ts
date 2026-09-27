/** Minimal HiDPI line chart on a 2D canvas. */
export interface ChartSeries {
  t: number[];
  v: number[];
  color: string;
  width?: number;
  alpha?: number;
}

export class LineChart {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  yLabel = '';
  xLabel = '';
  fixedRange: [number, number] | null = null;

  constructor(parent: HTMLElement, height = 110) {
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'chart';
    this.canvas.style.height = `${height}px`;
    parent.append(this.canvas);
    this.ctx = this.canvas.getContext('2d')!;
  }

  draw(series: ChartSeries[]) {
    const c = this.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(10, Math.floor(c.clientWidth * dpr));
    const hgt = Math.max(10, Math.floor(c.clientHeight * dpr));
    if (c.width !== w || c.height !== hgt) { c.width = w; c.height = hgt; }
    const g = this.ctx;
    g.clearRect(0, 0, w, hgt);
    let t0 = Infinity, t1 = -Infinity, v0 = Infinity, v1 = -Infinity;
    for (const s of series) {
      for (let i = 0; i < s.t.length; i++) {
        t0 = Math.min(t0, s.t[i]); t1 = Math.max(t1, s.t[i]);
        if (isFinite(s.v[i])) { v0 = Math.min(v0, s.v[i]); v1 = Math.max(v1, s.v[i]); }
      }
    }
    // tiny sparklines (probes) skip the axis furniture
    const compact = c.clientHeight < 50;
    const padL = (compact ? 2 : 34) * dpr, padR = (compact ? 2 : 6) * dpr, padT = (compact ? 3 : 6) * dpr, padB = (compact ? 3 : 14) * dpr;
    g.font = `${10 * dpr}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    g.fillStyle = 'rgba(160,172,190,0.7)';
    if (!isFinite(t0) || t1 <= t0) {
      g.fillText('waiting for data…', padL, hgt / 2);
      return;
    }
    if (this.fixedRange) [v0, v1] = this.fixedRange;
    if (v1 - v0 < 1e-6) { v0 -= 0.05; v1 += 0.05; }
    const pad = (v1 - v0) * 0.1;
    v0 -= pad; v1 += pad;
    const X = (t: number) => padL + ((t - t0) / (t1 - t0)) * (w - padL - padR);
    const Y = (v: number) => padT + (1 - (v - v0) / (v1 - v0)) * (hgt - padT - padB);
    // grid + labels
    g.strokeStyle = 'rgba(120,135,160,0.18)';
    g.lineWidth = 1;
    const ticks = compact ? [] : niceTicks(v0, v1, 4);
    for (const tv of ticks) {
      const y = Math.round(Y(tv)) + 0.5;
      g.beginPath(); g.moveTo(padL, y); g.lineTo(w - padR, y); g.stroke();
      g.fillText(fmt(tv), 2 * dpr, y + 3 * dpr);
    }
    if (this.xLabel && !compact) g.fillText(this.xLabel, w - padR - g.measureText(this.xLabel).width, hgt - 2 * dpr);
    for (const s of series) {
      g.strokeStyle = s.color;
      g.globalAlpha = s.alpha ?? 1;
      g.lineWidth = (s.width ?? 1.5) * dpr;
      g.beginPath();
      let started = false;
      for (let i = 0; i < s.t.length; i++) {
        if (!isFinite(s.v[i])) continue;
        const x = X(s.t[i]), y = Y(Math.min(Math.max(s.v[i], v0), v1));
        if (!started) { g.moveTo(x, y); started = true; } else g.lineTo(x, y);
      }
      g.stroke();
      g.globalAlpha = 1;
    }
  }
}

function niceTicks(a: number, b: number, n: number): number[] {
  const span = b - a;
  const raw = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-12; v += step) out.push(v);
  return out;
}

function fmt(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(2);
}
