/**
 * Lattice ↔ SI unit mapping.
 *
 * The lattice uses Δx = Δt = 1 and ρ₀ = 1. A simulation is characterised by
 *   L  – reference length in cells (vehicle length, or diameter for validation bodies)
 *   U  – inlet speed in cells/step (kept ≤ 0.08 so Ma = U/c_s ≤ 0.14, c_s = 1/√3)
 *   ν  – lattice viscosity, from which τ = 3ν + ½
 * and the only thing it shares with the real flow is the Reynolds number Re = U·L/ν.
 */
export const AIR_RHO = 1.225; // kg/m³
export const AIR_NU = 1.5e-5; // m²/s at 20 °C
export const CS = 1 / Math.sqrt(3);
export const U_MAX_LATTICE = 0.12;

export interface FlowSetup {
  /** real wind speed, m/s */
  speed: number;
  /** real reference length, m */
  lengthM: number;
  /** kinematic viscosity of the fluid, m²/s */
  nuAir: number;
  /** reference length in lattice cells */
  Lcells: number;
  /** maximum wind speed on the slider, m/s (sets the Reynolds scale) */
  vMax: number;
  /** smallest molecular lattice viscosity the grid is allowed to use */
  nuMin: number;
  /** optional user override of the simulated Reynolds number */
  reOverride?: number | null;
  /** multiplier from the stability watchdog (≤ 1), lowers U */
  uScale: number;
  /** multiplier from the stability watchdog (≥ 1), raises ν */
  nuScale: number;
}

export interface FlowMapping {
  reReal: number;
  reSim: number;
  U: number;
  nu: number;
  tau: number;
  mach: number;
  /** metres per cell */
  dx: number;
  /** seconds per step */
  dt: number;
}

/** Lattice inlet speed for a given real speed: grows with speed so the flow visibly speeds up. */
export function latticeSpeed(speed: number, vMax: number): number {
  const s = Math.min(Math.max(speed / vMax, 0), 1);
  return 0.03 + (0.08 - 0.03) * s;
}

/**
 * Map a real configuration onto the lattice. Real vehicle Reynolds numbers (10⁶–10⁷) are far
 * above what any affordable grid resolves, so the simulated Re is the real Re scaled by a fixed
 * factor chosen such that the top speed reaches the grid's resolvable limit. Dynamic similarity
 * therefore holds *within* the simulator (doubling speed doubles Re_sim), while the LES model
 * accounts for the unresolved scales.
 */
export function mapFlow(s: FlowSetup): FlowMapping {
  const reReal = (s.speed * s.lengthM) / s.nuAir;
  const U = latticeSpeed(s.speed, s.vMax) * s.uScale;
  const reMaxGrid = (0.08 * s.Lcells) / s.nuMin;
  const reRealMax = (s.vMax * s.lengthM) / AIR_NU;
  let reSim = s.reOverride && s.reOverride > 0 ? s.reOverride : reReal * (reMaxGrid / reRealMax);
  let nu = (U * s.Lcells) / reSim;
  nu = Math.max(nu * s.nuScale, s.nuMin * 0.5);
  reSim = (U * s.Lcells) / nu;
  const dx = s.lengthM / s.Lcells;
  const dt = s.speed > 0 ? (U / s.speed) * dx : 0;
  return { reReal, reSim, U, nu, tau: 3 * nu + 0.5, mach: U / CS, dx, dt };
}

export function formatRe(re: number): string {
  if (!isFinite(re)) return '—';
  if (re >= 1e6) return (re / 1e6).toFixed(2) + '·10⁶';
  if (re >= 1e4) return (re / 1e3).toFixed(0) + 'k';
  if (re >= 1e3) return (re / 1e3).toFixed(1) + 'k';
  return re.toFixed(0);
}
