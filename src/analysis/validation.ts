import type { SolverFactory } from '../backend/types';
import { GridDims } from '../solver/types';
import { coefficients } from './aero';
import { crossingFrequency, cylinderStRef, sphereCdRef } from './strouhal';

export interface Metric {
  label: string;
  value: string;
  expected?: string;
}

export interface ValidationResult {
  id: string;
  name: string;
  passed: boolean;
  metrics: Metric[];
  /** optional time series for plotting (e.g. lift coefficient) */
  series?: { t: number[]; v: number[]; label: string };
  error?: string;
  seconds: number;
}

export interface ValidationCase {
  id: string;
  name: string;
  description: string;
  needs3D: boolean;
  run(make: SolverFactory, progress: (f: number, msg?: string) => void, abort: () => boolean): Promise<ValidationResult>;
}

/** Analytic obstacle flags for a circle (2D) or sphere (3D) — exact, no mesh involved. */
function roundFlags(d: GridDims, c: [number, number, number], r: number, cylinder: boolean): { flags: Uint32Array; frontal: number } {
  const f = new Uint32Array(d.nx * d.ny * d.nz);
  const proj = new Uint8Array(d.ny * d.nz);
  for (let z = 0; z < d.nz; z++)
    for (let y = 0; y < d.ny; y++)
      for (let x = 0; x < d.nx; x++) {
        const dx = x + 0.5 - c[0], dy = y + 0.5 - c[1], dz = cylinder ? 0 : z + 0.5 - c[2];
        if (dx * dx + dy * dy + dz * dz < r * r) {
          f[x + d.nx * (y + d.ny * z)] = 1;
          proj[y + d.ny * z] = 1;
        }
      }
  let frontal = 0;
  for (const p of proj) frontal += p;
  return { flags: f, frontal };
}

const nextFrame = () => new Promise((r) => setTimeout(r, 0));

/** Scale factor for quick runs in slow environments (?valscale=0.5). */
const scaleParam = typeof location !== 'undefined' ? parseFloat(new URLSearchParams(location.search).get('valscale') ?? '1') : 1;

export const CASES: ValidationCase[] = [
  {
    id: 'freestream',
    name: 'Free-stream uniformity',
    description: 'Empty tunnel: the uniform inflow must stay uniform (no spurious currents, mass conserved).',
    needs3D: false,
    async run(make, progress) {
      const t0 = performance.now();
      const d = { nx: 256, ny: 96, nz: 1 };
      const U = 0.08;
      const s = make(d, { u: U, nu: 0.01, cs: 0.1, ground: 'freeslip', sides: 'freeslip', spongeNu: 0.05, spongeStart: 0.85, emaAlpha: 0.02 });
      s.uploadFlags(new Uint32Array(d.nx * d.ny));
      let st = await s.stepAndSample(1);
      for (let i = 0; i < 20; i++) {
        st = await s.stepAndSample(50);
        progress((i + 1) / 20);
      }
      s.destroy();
      const err = Math.abs(st.maxU - U) / U;
      const drho = Math.abs(st.rhoRef - 1);
      return {
        id: 'freestream', name: this.name, passed: err < 0.01 && drho < 0.01 && !st.unstable, seconds: (performance.now() - t0) / 1000,
        metrics: [
          { label: 'max |u| / U', value: (st.maxU / U).toFixed(4), expected: '1 ± 0.01' },
          { label: 'ρ at inlet plane', value: st.rhoRef.toFixed(4), expected: '1 ± 0.01' },
        ],
      };
    },
  },
  {
    id: 'cylinder',
    name: 'Cylinder, Re = 100 (von Kármán street)',
    description: 'D2Q9, D = 24 cells, 10% blockage. Strouhal number from the lift-coefficient oscillation.',
    needs3D: false,
    async run(make, progress, abort) {
      const t0 = performance.now();
      const k = scaleParam;
      const D = Math.round(24 * k);
      const d = { nx: Math.round(25 * D), ny: 10 * D, nz: 1 };
      const U = 0.06;
      const Re = 100;
      const nu = (U * D) / Re;
      const s = make(d, { u: U, nu, cs: 0, ground: 'freeslip', sides: 'freeslip', spongeNu: 0.03, spongeStart: 0.85, emaAlpha: 0.0 });
      const { flags } = roundFlags(d, [6 * D, d.ny / 2 + 0.37, 0], D / 2, true);
      s.uploadFlags(flags);
      const period = D / (0.17 * U);
      const total = Math.round(period * 16);
      const measureFrom = Math.round(period * 6);
      const chunk = Math.max(5, Math.round(period / 40));
      const tt: number[] = [], cl: number[] = [], cd: number[] = [];
      for (let step = 0; step < total; step += chunk) {
        const st = await s.stepAndSample(chunk);
        if (st.unstable) break;
        const c = coefficients(st.fx, st.fy, 0, U, D);
        if (step >= measureFrom) { tt.push(st.step); cl.push(c.cl); cd.push(c.cd); }
        if ((step / chunk) % 20 === 0) { progress(step / total, `step ${step}/${total}`); await nextFrame(); }
        if (abort()) break;
      }
      s.destroy();
      const { f, cycles, amplitude } = crossingFrequency(tt, cl);
      const St = (f * D) / U;
      const cdMean = cd.reduce((a, b) => a + b, 0) / Math.max(cd.length, 1);
      const passed = St >= 0.16 && St <= 0.2 && cycles >= 4;
      return {
        id: 'cylinder', name: this.name, passed, seconds: (performance.now() - t0) / 1000,
        series: { t: tt, v: cl, label: 'C_L(t)' },
        metrics: [
          { label: 'Strouhal St = fD/U', value: isFinite(St) ? St.toFixed(4) : '—', expected: `0.16 – 0.20 (Williamson: ${cylinderStRef(Re).toFixed(3)})` },
          { label: 'mean C_D', value: cdMean.toFixed(3), expected: '≈ 1.3 – 1.5 (confined)' },
          { label: "C_L amplitude", value: amplitude.toFixed(3), expected: '≈ 0.3 – 0.4' },
          { label: 'shedding cycles measured', value: String(cycles) },
          { label: 'grid / τ', value: `${d.nx}×${d.ny}, τ = ${(3 * nu + 0.5).toFixed(4)}` },
        ],
      };
    },
  },
  {
    id: 'sphere',
    name: 'Sphere, Re = 100 (drag)',
    description: 'D3Q19, D = 20 cells. Mean drag coefficient vs. the Schiller–Naumann correlation.',
    needs3D: true,
    async run(make, progress, abort) {
      const t0 = performance.now();
      const k = scaleParam;
      const D = Math.round(20 * k);
      const d = { nx: Math.round(9.6 * D), ny: Math.round(4.8 * D), nz: Math.round(4.8 * D) };
      const U = 0.05;
      const Re = 100;
      const nu = (U * D) / Re;
      const s = make(d, { u: U, nu, cs: 0, ground: 'freeslip', sides: 'freeslip', spongeNu: 0.03, spongeStart: 0.85, emaAlpha: 0 });
      const { flags, frontal } = roundFlags(d, [2.8 * D, d.ny / 2 + 0.37, d.nz / 2 + 0.21], D / 2, false);
      s.uploadFlags(flags);
      const flowThrough = d.nx / U;
      const total = Math.round(flowThrough * 2.2);
      const measureFrom = Math.round(flowThrough * 1.4);
      const chunk = 50;
      const tt: number[] = [], cds: number[] = [];
      for (let step = 0; step < total; step += chunk) {
        const st = await s.stepAndSample(chunk);
        if (st.unstable) break;
        const c = coefficients(st.fx, st.fy, st.fz, U, frontal);
        tt.push(st.step);
        cds.push(c.cd);
        if ((step / chunk) % 4 === 0) { progress(step / total, `step ${step}/${total}`); await nextFrame(); }
        if (abort()) break;
      }
      s.destroy();
      const sel = cds.filter((_, i) => tt[i] >= measureFrom);
      const cd = sel.reduce((a, b) => a + b, 0) / Math.max(sel.length, 1);
      const ref = sphereCdRef(Re);
      const err = (cd - ref) / ref;
      return {
        id: 'sphere', name: this.name, passed: Math.abs(err) < 0.2, seconds: (performance.now() - t0) / 1000,
        series: { t: tt, v: cds, label: 'C_D(t)' },
        metrics: [
          { label: 'mean C_D', value: isFinite(cd) ? cd.toFixed(3) : '—', expected: `${ref.toFixed(3)} ± 20 % (Schiller–Naumann)` },
          { label: 'deviation', value: isFinite(err) ? (err * 100).toFixed(1) + ' %' : '—' },
          { label: 'grid / τ', value: `${d.nx}×${d.ny}×${d.nz}, τ = ${(3 * nu + 0.5).toFixed(4)}` },
          { label: 'blockage', value: ((frontal / (d.ny * d.nz)) * 100).toFixed(1) + ' %' },
        ],
      };
    },
  },
];
