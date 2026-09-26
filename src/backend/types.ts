import type { RenderState } from '../render/renderState';
import type { GridDims, SolverParams, SolverStats } from '../solver/types';
import type { MeshData } from '../voxelize/mesh';
import type { VoxelInfo } from '../voxelize/VoxelizerGPU';

/** Minimal solver surface used by the validation cases (implemented by both GPU solvers). */
export interface LbmSolver {
  readonly dims: GridDims;
  readonly n: number;
  stepCount: number;
  uploadFlags(flags: Uint32Array): void;
  stepAndSample(steps: number): Promise<SolverStats>;
  destroy(): void;
}

export type SolverFactory = (dims: GridDims, p: SolverParams) => LbmSolver;

/**
 * A compute + render backend. The app logic (units, averaging, stability, UI) is shared; the
 * backend owns the solver, the voxelizer and the renderer for one graphics API.
 */
export interface Backend {
  readonly kind: 'webgpu' | 'webgl2';
  readonly label: string;
  readonly supports3D: boolean;
  readonly canvas: HTMLCanvasElement;
  /** smoothed GPU time per frame (ms) */
  readonly gpuMs: number;
  readonly stepCount: number;
  readonly cells: number;
  fits(dims: GridDims): boolean;
  createSolver(dims: GridDims, p: SolverParams): Promise<void>;
  setParams(p: Partial<SolverParams>): void;
  reset(): void;
  clearStats(): void;
  voxelize(mesh: MeshData | null, matrix: ArrayLike<number>): Promise<VoxelInfo>;
  setMesh(mesh: MeshData | null): void;
  refillParticles(): void;
  pixelSize(): [number, number];
  /** true while too many frames are in flight — skip this frame */
  throttled(): boolean;
  /** advance the solver, update fields, draw. Statistics arrive asynchronously through onStats. */
  frame(steps: number, rs: RenderState, onStats: (s: SolverStats) => void): void;
  /** factory for validation solvers (fresh instances independent of the scene) */
  solverFactory(): SolverFactory;
}
