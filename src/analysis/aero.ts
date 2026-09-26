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
