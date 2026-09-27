import { RendererGL } from '../render/RendererGL';
import type { RenderState } from '../render/renderState';
import { SolverGL } from '../solver/SolverGL';
import type { GridDims, SolverParams, SolverStats } from '../solver/types';
import type { MeshData } from '../voxelize/mesh';
import { voxelizeCPU } from '../voxelize/voxelizeCPU';
import type { VoxelInfo } from '../voxelize/VoxelizerGPU';
import type { Backend, SolverFactory } from './types';

/** WebGL2 fallback: fragment-shader D2Q9 solver, CPU voxelizer, Three.js renderer. 2D only. */
export class WebGLBackend implements Backend {
  readonly kind = 'webgl2' as const;
  readonly supports3D = false;
  readonly renderer: RendererGL;
  solver: SolverGL | null = null;
  gpuMs = 8;
  private lastT = performance.now();
  private maxTex: number;

  constructor(readonly canvas: HTMLCanvasElement) {
    this.renderer = new RendererGL(canvas);
    const gl = this.renderer.gl;
    this.maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('This browser supports neither WebGPU nor float render targets in WebGL2.');
  }

  get label() {
    const gl = this.renderer.gl;
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    return dbg ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)) : 'WebGL2';
  }
  get stepCount() {
    return this.solver?.stepCount ?? 0;
  }
  get cells() {
    return this.solver?.n ?? 0;
  }

  fits(d: GridDims): boolean {
    return d.nz === 1 && d.nx <= this.maxTex && d.ny <= this.maxTex;
  }

  async createSolver(dims: GridDims, p: SolverParams) {
    this.solver?.destroy();
    this.solver = new SolverGL(this.renderer.gl, dims, p);
    this.renderer.setSolver(this.solver);
  }
  setParams(p: Partial<SolverParams>) {
    this.solver?.setParams(p);
  }
  reset() {
    this.solver?.reset();
    this.renderer.refillParticles();
  }
  clearStats() {
    this.solver?.clearStats();
  }
  async voxelize(mesh: MeshData | null, matrix: ArrayLike<number>): Promise<VoxelInfo> {
    const s = this.solver!;
    const { flags, info, sdf } = voxelizeCPU(mesh, matrix, s.dims);
    s.uploadFlags(flags);
    s.uploadSdf(sdf);
    return info;
  }
  setMesh(mesh: MeshData | null) {
    this.renderer.setMesh(mesh);
  }
  refillParticles() {
    this.renderer.refillParticles();
  }
  pixelSize(): [number, number] {
    return this.renderer.pixelSize();
  }
  throttled() {
    return false;
  }

  frame(steps: number, rs: RenderState, onStats: (s: SolverStats) => void) {
    const s = this.solver!;
    if (steps > 0) s.step(steps);
    s.post(true);
    s.pollStats(onStats);
    this.renderer.render(rs);
    // No portable GPU timer in WebGL: while frames keep up with vsync report headroom (the step
    // tuner grows), once they don't, report the real interval (it shrinks).
    const now = performance.now();
    const interval = now - this.lastT;
    this.lastT = now;
    this.gpuMs = this.gpuMs * 0.7 + (interval < 1000 / 52 ? 6 : interval) * 0.3;
  }

  exportFields(_maxPoints: number) {
    return this.solver!.exportFields();
  }

  probeData() {
    return this.renderer.probeData;
  }

  solverFactory(): SolverFactory {
    return (dims, p) => new SolverGL(this.renderer.gl, dims, p);
  }
}
