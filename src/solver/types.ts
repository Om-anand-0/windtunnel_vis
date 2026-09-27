export type GroundMode = 'freeslip' | 'noslip' | 'moving';

export interface SolverParams {
  /** lattice inlet velocity (cells / step) */
  u: number;
  /** molecular lattice viscosity */
  nu: number;
  /** Smagorinsky constant */
  cs: number;
  ground: GroundMode;
  /** z-boundaries in 3D: free slip or periodic */
  sides: 'freeslip' | 'periodic';
  /** extra viscosity in the outlet sponge */
  spongeNu: number;
  /** fraction of nx where the sponge starts */
  spongeStart: number;
  /** width (fraction of nx) of the absorbing layer behind the inlet */
  spongeIn?: number;
  /** minimum relaxation time on wall-adjacent cells (crude wall model, damps wall modes) */
  tauWall?: number;
  /** 'regularized' (default) or plain 'bgk' */
  collision?: 'regularized' | 'bgk';
  /** EMA weight for time averages at this frame */
  emaAlpha: number;
}

export interface SolverStats {
  /** force on the vehicle in lattice units (momentum exchange, averaged over a single step) */
  fx: number;
  fy: number;
  fz: number;
  maxU: number;
  /** mean density on the reference plane upstream */
  rhoRef: number;
  unstable: boolean;
  /** solver step at which the sample was taken */
  step: number;
}

/** Down-sampled field snapshot for export. All arrays are 4 floats per point, x fastest. */
export interface FieldExport {
  dims: [number, number, number];
  stride: number;
  /** (ux, uy, uz, ρ) */
  vel: Float32Array;
  /** time-averaged (ux, uy, uz, ρ) */
  mean: Float32Array;
  /** (ωx, ωy, ωz, Q) */
  vort: Float32Array;
}

export interface GridDims {
  nx: number;
  ny: number;
  nz: number;
}
