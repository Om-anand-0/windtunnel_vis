import type { GpuContext } from '../gpu/device';
import { RendererGPU } from '../render/RendererGPU';
import type { RenderState } from '../render/renderState';
import { SolverGPU } from '../solver/SolverGPU';
import type { GridDims, SolverParams, SolverStats } from '../solver/types';
import type { MeshData } from '../voxelize/mesh';
import { VoxelInfo, VoxelizerGPU } from '../voxelize/VoxelizerGPU';
import type { Backend, SolverFactory } from './types';

export class WebGPUBackend implements Backend {
  readonly kind = 'webgpu' as const;
  readonly supports3D = true;
  readonly renderer: RendererGPU;
  private vox: VoxelizerGPU;
  solver: SolverGPU | null = null;
  gpuMs = 8;
  private inFlight = 0;
  private lastDone = 0;

  // GPU timestamps (when available) measure the real per-frame GPU time for the step tuner
  private qs: GPUQuerySet | null = null;
  private qResolve: GPUBuffer | null = null;
  private qRead: { buf: GPUBuffer; busy: boolean }[] = [];

  constructor(readonly gpu: GpuContext, readonly canvas: HTMLCanvasElement) {
    this.renderer = new RendererGPU(gpu.device, canvas);
    this.vox = new VoxelizerGPU(gpu.device);
    if (gpu.hasTimestamps) {
      try {
        const d = gpu.device;
        this.qs = d.createQuerySet({ type: 'timestamp', count: 2 });
        this.qResolve = d.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
        for (let i = 0; i < 3; i++) this.qRead.push({ buf: d.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), busy: false });
      } catch {
        this.qs = null;
      }
    }
  }

  private stamp(enc: GPUCommandEncoder, index: 0 | 1) {
    const p = enc.beginComputePass({ timestampWrites: { querySet: this.qs!, ...(index === 0 ? { beginningOfPassWriteIndex: 0 } : { endOfPassWriteIndex: 1 }) } });
    p.end();
  }

  get label() {
    return this.gpu.adapterInfo;
  }
  get device() {
    return this.gpu.device;
  }
  get stepCount() {
    return this.solver?.stepCount ?? 0;
  }
  get cells() {
    return this.solver?.n ?? 0;
  }

  fits(d: GridDims): boolean {
    const need = d.nx * d.ny * d.nz * (d.nz > 1 ? 19 : 9) * 4;
    return need <= this.gpu.maxBinding && need <= this.gpu.maxBuffer;
  }

  async createSolver(dims: GridDims, p: SolverParams) {
    await this.device.queue.onSubmittedWorkDone();
    this.solver?.destroy();
    this.solver = new SolverGPU(this.device, dims, p);
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
  voxelize(mesh: MeshData | null, matrix: ArrayLike<number>): Promise<VoxelInfo> {
    return this.vox.voxelize(this.solver!, mesh, matrix);
  }
  setMesh(mesh: MeshData | null) {
    this.renderer.setMesh(mesh);
  }
  refillParticles() {
    this.renderer.refillParticles();
  }
  pixelSize(): [number, number] {
    return this.renderer.pixelSize;
  }
  throttled() {
    return this.inFlight >= 2;
  }

  frame(steps: number, rs: RenderState, onStats: (s: SolverStats) => void) {
    const s = this.solver!;
    const enc = this.device.createCommandEncoder();
    const rb = this.qs ? this.qRead.find((r) => !r.busy) : undefined;
    if (rb) this.stamp(enc, 0);
    if (steps > 0) s.encodeSteps(enc, steps);
    s.encodePost(enc);
    this.renderer.render(enc, rs);
    if (rb) {
      this.stamp(enc, 1);
      enc.resolveQuerySet(this.qs!, 0, 2, this.qResolve!, 0);
      enc.copyBufferToBuffer(this.qResolve!, 0, rb.buf, 0, 16);
      rb.busy = true;
    }
    const tSubmit = performance.now();
    this.device.queue.submit([enc.finish()]);
    this.inFlight++;
    s.flushReads(onStats);
    this.renderer.afterSubmit();
    if (rb) {
      rb.buf.mapAsync(GPUMapMode.READ).then(() => {
        const t = new BigInt64Array(rb.buf.getMappedRange().slice(0));
        rb.buf.unmap();
        rb.busy = false;
        const ms = Number(t[1] - t[0]) / 1e6;
        if (ms > 0 && ms < 1000) this.gpuMs = this.gpuMs * 0.8 + ms * 0.2;
      }).catch(() => (rb.busy = false));
    }
    this.device.queue.onSubmittedWorkDone().then(() => {
      this.inFlight--;
      const done = performance.now();
      const g = done - Math.max(tSubmit, this.lastDone);
      this.lastDone = done;
      if (!this.qs) this.gpuMs = this.gpuMs * 0.8 + g * 0.2;
    }).catch(() => (this.inFlight = 0));
  }

  exportFields(maxPoints: number) {
    const s = this.solver!;
    const stride = Math.max(1, Math.ceil(Math.cbrt(s.n / maxPoints)));
    return s.exportFields(stride);
  }

  probeData() {
    return this.renderer.probeData;
  }

  solverFactory(): SolverFactory {
    return (dims, p) => new SolverGPU(this.device, dims, p);
  }
}
