import { dispatch1D } from '../gpu/device';
import { D2Q9, D3Q19, Lattice } from './lattice';
import { GridDims, SolverParams, SolverStats } from './types';
import { derivedWGSL, initWGSL, macroWGSL, reduceWGSL, streamCollideWGSL } from './wgslLbm';

const WG = 128;

/**
 * GPU lattice Boltzmann solver (D2Q9 when nz == 1, D3Q19 otherwise).
 * Owns the population buffers, obstacle flags, statistics and the field textures used by the renderer.
 */
export class SolverGPU {
  readonly lattice: Lattice;
  readonly dims: GridDims;
  readonly n: number;
  readonly is3D: boolean;

  readonly fA: GPUBuffer;
  readonly fB: GPUBuffer;
  readonly flags: GPUBuffer;
  readonly meanA: GPUBuffer;
  readonly meanB: GPUBuffer;
  readonly velTex: GPUTexture;
  readonly vortTex: GPUTexture;
  readonly statTex: GPUTexture;
  readonly meanTex: GPUTexture;

  private params: GPUBuffer;
  private partials: GPUBuffer;
  private result: GPUBuffer;
  private reduceParams: GPUBuffer;
  private staging: { buf: GPUBuffer; busy: boolean }[] = [];

  private scPipe: GPUComputePipeline;
  private initPipe: GPUComputePipeline;
  private macroPipe: GPUComputePipeline;
  private reducePipe: GPUComputePipeline;
  private derivedPipe: GPUComputePipeline;
  private scBG: [GPUBindGroup, GPUBindGroup];
  private initBG: [GPUBindGroup, GPUBindGroup];
  private macroBG: [GPUBindGroup, GPUBindGroup];
  private reduceBG: GPUBindGroup;
  private derivedBG: GPUBindGroup;
  private disp: { x: number; y: number; strideX: number };
  private partialCount: number;

  /** 0: current populations are in A, 1: in B */
  private parity = 0;
  stepCount = 0;
  latest: SolverStats | null = null;
  private p: SolverParams;
  private pendingInit = false;

  constructor(readonly device: GPUDevice, dims: GridDims, params: SolverParams) {
    this.dims = dims;
    this.is3D = dims.nz > 1;
    this.lattice = this.is3D ? D3Q19 : D2Q9;
    this.n = dims.nx * dims.ny * dims.nz;
    this.p = { ...params };
    const q = this.lattice.q;
    const S = GPUBufferUsage.STORAGE;
    const fBytes = q * this.n * 4;
    this.fA = device.createBuffer({ size: fBytes, usage: S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, label: 'fA' });
    this.fB = device.createBuffer({ size: fBytes, usage: S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, label: 'fB' });
    this.flags = device.createBuffer({ size: this.n * 4, usage: S | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, label: 'flags' });
    this.meanA = device.createBuffer({ size: this.n * 16, usage: S | GPUBufferUsage.COPY_DST, label: 'meanA' });
    this.meanB = device.createBuffer({ size: this.n * 4, usage: S | GPUBufferUsage.COPY_DST, label: 'meanB' });
    this.params = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.disp = dispatch1D(this.n, WG);
    this.partialCount = this.disp.x * this.disp.y;
    this.partials = device.createBuffer({ size: this.partialCount * 32, usage: S });
    this.result = device.createBuffer({ size: 32, usage: S | GPUBufferUsage.COPY_SRC });
    this.reduceParams = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.reduceParams, 0, new Uint32Array([this.partialCount, 0, 0, 0]));
    for (let i = 0; i < 4; i++) {
      this.staging.push({ buf: device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), busy: false });
    }

    const tex = (label: string) =>
      device.createTexture({
        label,
        size: [dims.nx, dims.ny, dims.nz],
        dimension: '3d',
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });
    this.velTex = tex('vel');
    this.vortTex = tex('vort');
    this.statTex = tex('stat');
    this.meanTex = tex('mean');

    const mk = (code: string, label: string) =>
      device.createComputePipeline({
        label,
        layout: 'auto',
        compute: { module: device.createShaderModule({ code, label }), entryPoint: 'main' },
      });
    const L = this.lattice;
    this.scPipe = mk(streamCollideWGSL(L, WG), 'stream-collide');
    this.initPipe = mk(initWGSL(L, WG), 'init');
    this.macroPipe = mk(macroWGSL(L, WG), 'macro');
    this.reducePipe = mk(reduceWGSL(), 'reduce');
    this.derivedPipe = mk(derivedWGSL(this.is3D ? 3 : 2), 'derived');

    const bg = (pipe: GPUComputePipeline, entries: [number, GPUBindingResource][]) =>
      device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: entries.map(([binding, resource]) => ({ binding, resource })),
      });
    const B = (buffer: GPUBuffer) => ({ buffer });
    this.scBG = [
      bg(this.scPipe, [[0, B(this.params)], [1, B(this.fA)], [2, B(this.fB)], [3, B(this.flags)]]),
      bg(this.scPipe, [[0, B(this.params)], [1, B(this.fB)], [2, B(this.fA)], [3, B(this.flags)]]),
    ];
    this.initBG = [
      bg(this.initPipe, [[0, B(this.params)], [2, B(this.fA)], [3, B(this.flags)]]),
      bg(this.initPipe, [[0, B(this.params)], [2, B(this.fB)], [3, B(this.flags)]]),
    ];
    const velView = this.velTex.createView();
    this.macroBG = [this.fA, this.fB].map((f) =>
      bg(this.macroPipe, [
        [0, B(this.params)], [1, B(f)], [2, B(this.flags)], [3, velView],
        [4, B(this.meanA)], [5, B(this.meanB)], [6, B(this.partials)],
      ]),
    ) as [GPUBindGroup, GPUBindGroup];
    this.reduceBG = bg(this.reducePipe, [[0, B(this.reduceParams)], [1, B(this.partials)], [2, B(this.result)]]);
    this.derivedBG = bg(this.derivedPipe, [
      [0, B(this.params)], [1, velView], [2, B(this.flags)], [3, B(this.meanA)], [4, B(this.meanB)],
      [5, this.vortTex.createView()], [6, this.statTex.createView()], [7, this.meanTex.createView()],
    ]);
    this.writeParams();
    this.reset();
  }

  get params_(): SolverParams {
    return this.p;
  }

  /** Current tau0 (molecular relaxation time). */
  get tau0(): number {
    return 3 * this.p.nu + 0.5;
  }

  setParams(p: Partial<SolverParams>) {
    Object.assign(this.p, p);
    this.writeParams();
  }

  private writeParams() {
    const { nx, ny, nz } = this.dims;
    const buf = new ArrayBuffer(96);
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    const p = this.p;
    u[0] = nx; u[1] = ny; u[2] = nz; u[3] = this.n;
    f[4] = 3 * p.nu + 0.5;
    f[5] = p.cs * p.cs;
    f[6] = p.u;
    f[7] = 1.0;
    u[8] = p.ground === 'freeslip' ? 0 : p.ground === 'noslip' ? 1 : 2;
    u[9] = this.disp.strideX;
    f[10] = p.spongeStart * nx;
    f[11] = p.spongeNu;
    f[12] = p.ground === 'moving' ? p.u : 0;
    f[13] = p.emaAlpha;
    f[14] = Math.max(p.u, 1e-4);
    u[15] = Math.min(4, nx - 1);
    u[16] = p.sides === 'periodic' ? 1 : 0;
    u[17] = p.collision === 'bgk' ? 1 : 0;
    f[18] = p.tauWall ?? 0.5;
    f[20] = (p.spongeIn ?? 0) * nx;
    f[21] = 2.0;
    this.device.queue.writeBuffer(this.params, 0, buf);
  }

  /** Re-initialise populations to the uniform free-stream equilibrium and clear statistics. */
  reset() {
    this.pendingInit = true;
    this.stepCount = 0;
    this.clearStats();
  }

  clearStats() {
    this.device.queue.writeBuffer(this.meanB, 0, new Float32Array(this.n));
    const m = new Float32Array(this.n * 4);
    for (let i = 3; i < m.length; i += 4) m[i] = 1;
    this.device.queue.writeBuffer(this.meanA, 0, m);
  }

  /** Record `steps` fused stream-collide steps. */
  encodeSteps(enc: GPUCommandEncoder, steps: number) {
    const pass = enc.beginComputePass({ label: 'lbm' });
    if (this.pendingInit) {
      pass.setPipeline(this.initPipe);
      for (const g of this.initBG) {
        pass.setBindGroup(0, g);
        pass.dispatchWorkgroups(this.disp.x, this.disp.y);
      }
      this.pendingInit = false;
    }
    pass.setPipeline(this.scPipe);
    for (let s = 0; s < steps; s++) {
      pass.setBindGroup(0, this.scBG[this.parity]);
      pass.dispatchWorkgroups(this.disp.x, this.disp.y);
      this.parity ^= 1;
    }
    this.stepCount += steps;
    pass.end();
  }

  /** Macroscopic fields, statistics and force reduction. */
  encodePost(enc: GPUCommandEncoder, derived = true) {
    const { nx, ny, nz } = this.dims;
    const pass = enc.beginComputePass({ label: 'post' });
    if (this.pendingInit) {
      pass.setPipeline(this.initPipe);
      for (const g of this.initBG) {
        pass.setBindGroup(0, g);
        pass.dispatchWorkgroups(this.disp.x, this.disp.y);
      }
      this.pendingInit = false;
    }
    pass.setPipeline(this.macroPipe);
    pass.setBindGroup(0, this.macroBG[this.parity]);
    pass.dispatchWorkgroups(this.disp.x, this.disp.y);
    pass.setPipeline(this.reducePipe);
    pass.setBindGroup(0, this.reduceBG);
    pass.dispatchWorkgroups(1);
    if (derived) {
      pass.setPipeline(this.derivedPipe);
      pass.setBindGroup(0, this.derivedBG);
      pass.dispatchWorkgroups(Math.ceil(nx / 8), Math.ceil(ny / 8), Math.ceil(nz / (this.is3D ? 4 : 1)));
    }
    pass.end();
    if (!derived) return;
    const st = this.staging.find((s) => !s.busy);
    if (st) {
      enc.copyBufferToBuffer(this.result, 0, st.buf, 0, 32);
      st.busy = true;
      const step = this.stepCount;
      // mapping is kicked off after submit by readStats()
      this.pendingReads.push({ st, step });
    }
  }

  private pendingReads: { st: { buf: GPUBuffer; busy: boolean }; step: number }[] = [];

  /** Call after queue.submit(); resolves stats asynchronously into `latest`. */
  flushReads(onStats?: (s: SolverStats) => void) {
    const reads = this.pendingReads;
    this.pendingReads = [];
    for (const { st, step } of reads) {
      st.buf
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          const d = new Float32Array(st.buf.getMappedRange().slice(0));
          st.buf.unmap();
          st.busy = false;
          const nref = Math.max(d[5], 1);
          const s: SolverStats = {
            fx: d[0], fy: d[1], fz: d[2],
            maxU: Math.sqrt(Math.max(d[3], 0)),
            rhoRef: d[5] > 0 ? d[4] / nref : 1,
            unstable: d[6] > 0 || !isFinite(d[0]) || d[3] > 0.25,
            step,
          };
          if (!this.latest || step >= this.latest.step) this.latest = s;
          onStats?.(s);
        })
        .catch(() => {
          st.busy = false;
        });
    }
  }

  private sampleBuf: GPUBuffer | null = null;

  /** Run `steps` steps, then synchronously (awaited) read the force/statistics result. */
  async stepAndSample(steps: number): Promise<SolverStats> {
    const enc = this.device.createCommandEncoder();
    this.encodeSteps(enc, steps);
    this.encodePost(enc, false);
    if (!this.sampleBuf) this.sampleBuf = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(this.result, 0, this.sampleBuf, 0, 32);
    this.device.queue.submit([enc.finish()]);
    await this.sampleBuf.mapAsync(GPUMapMode.READ);
    const d = new Float32Array(this.sampleBuf.getMappedRange().slice(0));
    this.sampleBuf.unmap();
    return {
      fx: d[0], fy: d[1], fz: d[2],
      maxU: Math.sqrt(Math.max(d[3], 0)),
      rhoRef: d[5] > 0 ? d[4] / d[5] : 1,
      unstable: d[6] > 0 || !isFinite(d[0]) || d[3] > 0.25,
      step: this.stepCount,
    };
  }

  /** Upload obstacle flags from the CPU (used by validation cases and the fallback voxelizer). */
  uploadFlags(flags: Uint32Array) {
    this.device.queue.writeBuffer(this.flags, 0, flags);
  }

  /** Read back the flags buffer (debug / frontal area). */
  async readFlags(): Promise<Uint32Array> {
    const buf = this.device.createBuffer({ size: this.n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.flags, 0, buf, 0, this.n * 4);
    this.device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(buf.getMappedRange().slice(0));
    buf.destroy();
    return out;
  }

  /** Debug: read the current populations back to the CPU. */
  async readPopulations(): Promise<Float32Array> {
    const size = this.lattice.q * this.n * 4;
    const buf = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.current, 0, buf, 0, size);
    this.device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(buf.getMappedRange().slice(0));
    buf.destroy();
    return out;
  }

  /** Current population buffer (post-collision of the last step). */
  get current(): GPUBuffer {
    return this.parity === 0 ? this.fA : this.fB;
  }

  destroy() {
    for (const b of [this.fA, this.fB, this.flags, this.meanA, this.meanB, this.params, this.partials, this.result, this.reduceParams]) b.destroy();
    for (const s of this.staging) s.buf.destroy();
    this.sampleBuf?.destroy();
    for (const t of [this.velTex, this.vortTex, this.statTex, this.meanTex]) t.destroy();
  }

  static bytesFor(dims: GridDims): number {
    const n = dims.nx * dims.ny * dims.nz;
    const q = dims.nz > 1 ? 19 : 9;
    return n * (q * 8 + 4 + 20 + 32);
  }
}
