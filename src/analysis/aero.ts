import { AIR_RHO } from './units';

/** Aerodynamic coefficients from a lattice force. In 2D `area` is the frontal height in cells. */
export function coefficients(fx: number, fy: number, fz: number, U: number, area: number, rho = 1) {
  const q = 0.5 * rho * U * U * Math.max(area, 1e-9);
  return { cd: fx / q, cl: fy / q, cs: fz / q };
}

/** Real drag force in newtons at the chosen speed. */
export function dragNewtons(cd: number, speed: number, areaM2: number): number {
  return 0.5 * AIR_RHO * speed * speed * cd * areaM2;
}

/** Simple exponential moving average helper. */
export class Ema {
  value = NaN;
  constructor(public k: number) {}
  push(v: number) {
    if (!isFinite(v)) return this.value;
    this.value = isNaN(this.value) ? v : this.value + this.k * (v - this.value);
    return this.value;
  }
  reset() {
    this.value = NaN;
  }
}

/**
 * Mean and standard error of a correlated signal over its last `window` time units, by the
 * method of batch means (8 contiguous batches; valid when a batch is longer than the correlation
 * time — here a fraction of a convective time).
 */
export function windowStats(t: number[], v: number[], window: number) {
  const n = t.length;
  if (n < 16) return { mean: NaN, se: NaN, span: 0, n };
  const tEnd = t[n - 1];
  let i0 = n - 1;
  while (i0 > 0 && t[i0 - 1] >= tEnd - window) i0--;
  const span = tEnd - t[i0];
  const B = 8;
  const sums = new Array(B).fill(0), cnt = new Array(B).fill(0);
  let tot = 0, totN = 0;
  for (let i = i0; i < n; i++) {
    const b = Math.min(B - 1, Math.floor(((t[i] - t[i0]) / Math.max(span, 1e-9)) * B));
    sums[b] += v[i]; cnt[b]++; tot += v[i]; totN++;
  }
  const mean = tot / totN;
  const bm = sums.map((s, k) => (cnt[k] ? s / cnt[k] : NaN)).filter((x) => isFinite(x));
  if (bm.length < 4) return { mean, se: NaN, span, n: totN };
  const varB = bm.reduce((a, x) => a + (x - mean) ** 2, 0) / (bm.length - 1);
  return { mean, se: Math.sqrt(varB / bm.length), span, n: totN };
}

/** Fixed-capacity time series (step, value). */
export class Series {
  t: number[] = [];
  v: number[] = [];
  constructor(public cap = 600) {}
  push(t: number, v: number) {
    if (!isFinite(v)) return;
    this.t.push(t);
    this.v.push(v);
    if (this.t.length > this.cap) {
      this.t.shift();
      this.v.shift();
    }
  }
  clear() {
    this.t = [];
    this.v = [];
  }
  meanSince(t0: number): number {
    let s = 0, n = 0;
    for (let i = 0; i < this.t.length; i++) if (this.t[i] >= t0) { s += this.v[i]; n++; }
    return n ? s / n : NaN;
  }
}
