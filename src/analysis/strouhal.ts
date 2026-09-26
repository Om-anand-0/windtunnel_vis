/**
 * Dominant frequency of a (possibly unevenly sampled) signal from its mean-crossings.
 * Returns cycles per time unit, or NaN if fewer than 3 full cycles are available.
 */
export function crossingFrequency(t: number[], v: number[]): { f: number; cycles: number; amplitude: number } {
  const n = v.length;
  if (n < 8) return { f: NaN, cycles: 0, amplitude: NaN };
  let mean = 0;
  for (const x of v) mean += x;
  mean /= n;
  let lo = Infinity, hi = -Infinity;
  for (const x of v) { lo = Math.min(lo, x); hi = Math.max(hi, x); }
  const amp = (hi - lo) / 2;
  // hysteresis to ignore noise-induced chatter around the mean
  const hyst = amp * 0.15;
  const ups: number[] = [];
  let armed = v[0] < mean - hyst;
  for (let i = 1; i < n; i++) {
    if (v[i] < mean - hyst) armed = true;
    if (armed && v[i - 1] < mean && v[i] >= mean) {
      const a = (mean - v[i - 1]) / (v[i] - v[i - 1]);
      ups.push(t[i - 1] + a * (t[i] - t[i - 1]));
      armed = false;
    }
  }
  if (ups.length < 4) return { f: NaN, cycles: Math.max(0, ups.length - 1), amplitude: amp };
  const cycles = ups.length - 1;
  const period = (ups[ups.length - 1] - ups[0]) / cycles;
  return { f: 1 / period, cycles, amplitude: amp };
}

/** Schiller–Naumann sphere drag correlation, valid for Re < 800. */
export function sphereCdRef(re: number): number {
  return (24 / re) * (1 + 0.15 * Math.pow(re, 0.687));
}

/** Empirical Strouhal number for a circular cylinder (Williamson 1996 fit, 50 < Re < 180). */
export function cylinderStRef(re: number): number {
  return 0.2175 - 5.106 / re;
}
