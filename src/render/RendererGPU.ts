import type { SolverGPU } from '../solver/SolverGPU';
import type { MeshData } from '../voxelize/mesh';
import { PARTICLES_WGSL, STREAMLINES_WGSL } from './tracers';
import { GROUND_WGSL, MESH_WGSL, SEGMENTS_WGSL, SLICE_WGSL, STREAM_WGSL, TRAILS_WGSL, VOLUME_WGSL } from './wgslRender';
import type { RenderState } from './renderState';

const MSAA = 4;
/** debug: ?skip=ground,mesh,slice,volume,stream,trails,segments,particles */
/** debug: ?offscreen renders into a texture and blits through a readback (for headless testing) */
const OFFSCREEN = typeof location !== 'undefined' && new URLSearchParams(location.search).has('offscreen');
const SKIP = new Set((typeof location !== 'undefined' ? new URLSearchParams(location.search).get('skip') ?? '' : '').split(','));
const DEPTH = 'depth24plus';

interface TracerBuffers {
  count: number;
  trail: number;
  state: GPUBuffer;
  hist: GPUBuffer;
  computeBG: GPUBindGroup;
}

export class RendererGPU {
  readonly ctx: GPUCanvasContext | null = null;
  readonly format: GPUTextureFormat;
  private offTex: GPUTexture | null = null;
  private off2D: CanvasRenderingContext2D | null = null;
  private offBuf: GPUBuffer | null = null;
  private offBusy = false;
  private offFrame = 0;
  private offStore: GPUBuffer | null = null;
  private offPipe: GPUComputePipeline | null = null;
  private offBG: GPUBindGroup | null = null;
  private msaaTex: GPUTexture | null = null;
  private depthTex: GPUTexture | null = null;
  private size = [0, 0];

  private frameBuf: GPUBuffer;
  private frameLayout: GPUBindGroupLayout;
  private frameBG: GPUBindGroup | null = null;
  private sampler: GPUSampler;
  private solver: SolverGPU | null = null;

  private pipes = new Map<string, GPURenderPipeline>();
  private meshPipeLayout: GPUPipelineLayout;
  private meshU: GPUBuffer;
  private meshBG: GPUBindGroup;
  private meshVB: GPUBuffer | null = null;
  private meshIB: GPUBuffer | null = null;
  private meshCount = 0;

  private sliceU: GPUBuffer;
  private sliceBG: GPUBindGroup;
  private volU: GPUBuffer;
  private volBG: GPUBindGroup;
  private groundU: GPUBuffer;
  private groundBG: GPUBindGroup;
  private segBuf: GPUBuffer;
  private segBG: GPUBindGroup;
  private segCount = 0;

  private partPipe: GPUComputePipeline;
  private partU: GPUBuffer;
  private tracers: TracerBuffers | null = null;
  private trailHead = 0;
  private needFill = true;
  private frameSeed = 1;

  private slPipe: GPUComputePipeline;
  private slU: GPUBuffer;
  private slBuf: GPUBuffer | null = null;
  private slBG: GPUBindGroup | null = null;
  private slRenderBG: GPUBindGroup | null = null;
  private slSize = [0, 0];

  private layouts: Record<string, GPUBindGroupLayout> = {};

  constructor(readonly device: GPUDevice, readonly canvas: HTMLCanvasElement) {
    if (OFFSCREEN) {
      this.format = 'rgba8unorm';
      this.off2D = canvas.getContext('2d');
    } else {
      this.ctx = canvas.getContext('webgpu') as GPUCanvasContext;
      this.format = navigator.gpu.getPreferredCanvasFormat();
      this.ctx.configure({ device, format: this.format, alphaMode: 'opaque' });
    }
    const V = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE;
    this.frameLayout = device.createBindGroupLayout({
      label: 'frame',
      entries: [
        { binding: 0, visibility: V, buffer: { type: 'uniform' } },
        { binding: 1, visibility: V, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 2, visibility: V, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 3, visibility: V, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 4, visibility: V, sampler: { type: 'filtering' } },
        { binding: 5, visibility: V, buffer: { type: 'read-only-storage' } },
        { binding: 6, visibility: V, texture: { sampleType: 'float', viewDimension: '3d' } },
      ],
    });
    this.frameBuf = device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });

    const uni = (size: number) => device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const layout1 = (entries: GPUBindGroupLayoutEntry[]) => device.createBindGroupLayout({ entries });
    const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    this.layouts.uni = layout1([{ binding: 0, visibility: VF, buffer: { type: 'uniform' } }]);
    this.layouts.seg = layout1([{ binding: 0, visibility: VF, buffer: { type: 'read-only-storage' } }]);
    this.layouts.ribbon = layout1([
      { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
      { binding: 1, visibility: VF, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: VF, buffer: { type: 'read-only-storage' } },
    ]);
    this.layouts.stream = layout1([
      { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
      { binding: 1, visibility: VF, buffer: { type: 'read-only-storage' } },
    ]);
    const C = GPUShaderStage.COMPUTE;
    this.layouts.partC = layout1([
      { binding: 0, visibility: C, buffer: { type: 'uniform' } },
      { binding: 1, visibility: C, buffer: { type: 'storage' } },
      { binding: 2, visibility: C, buffer: { type: 'storage' } },
    ]);
    this.layouts.slC = layout1([
      { binding: 0, visibility: C, buffer: { type: 'uniform' } },
      { binding: 1, visibility: C, buffer: { type: 'storage' } },
    ]);
    this.meshPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout, this.layouts.uni] });

    this.meshU = uni(160);
    this.meshBG = device.createBindGroup({ layout: this.layouts.uni, entries: [{ binding: 0, resource: { buffer: this.meshU } }] });
    this.sliceU = uni(48);
    this.sliceBG = device.createBindGroup({ layout: this.layouts.uni, entries: [{ binding: 0, resource: { buffer: this.sliceU } }] });
    this.volU = uni(32);
    this.volBG = device.createBindGroup({ layout: this.layouts.uni, entries: [{ binding: 0, resource: { buffer: this.volU } }] });
    this.groundU = uni(32);
    this.groundBG = device.createBindGroup({ layout: this.layouts.uni, entries: [{ binding: 0, resource: { buffer: this.groundU } }] });
    this.segBuf = device.createBuffer({ size: 48 * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.segBG = device.createBindGroup({ layout: this.layouts.seg, entries: [{ binding: 0, resource: { buffer: this.segBuf } }] });

    this.partU = uni(80);
    this.partPipe = device.createComputePipeline({
      label: 'particles',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout, this.layouts.partC] }),
      compute: { module: device.createShaderModule({ code: PARTICLES_WGSL, label: 'particles' }), entryPoint: 'main' },
    });
    this.slU = uni(64);
    this.slPipe = device.createComputePipeline({
      label: 'streamlines',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout, this.layouts.slC] }),
      compute: { module: device.createShaderModule({ code: STREAMLINES_WGSL, label: 'streamlines' }), entryPoint: 'main' },
    });
  }

  private modules = new Map<string, GPUShaderModule>();
  private module(code: string, label: string) {
    let m = this.modules.get(label);
    if (!m) {
      m = this.device.createShaderModule({ code, label });
      this.modules.set(label, m);
    }
    return m;
  }

  private pipeline(key: string, make: () => GPURenderPipeline) {
    let p = this.pipes.get(key);
    if (!p) {
      p = make();
      this.pipes.set(key, p);
    }
    return p;
  }

  private basicPipe(label: string, code: string, group1: GPUBindGroupLayout, opts: {
    blend?: 'premul' | 'none'; depthWrite?: boolean; depthCompare?: GPUCompareFunction; cull?: GPUCullMode; topology?: GPUPrimitiveTopology;
    vertexBuffers?: GPUVertexBufferLayout[];
  }): GPURenderPipeline {
    const blend: GPUBlendState | undefined =
      opts.blend === 'premul'
        ? { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } }
        : undefined;
    const module = this.module(code, label);
    return this.device.createRenderPipeline({
      label,
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.frameLayout, group1] }),
      vertex: { module, entryPoint: 'vs', buffers: opts.vertexBuffers ?? [] },
      fragment: { module, entryPoint: 'fs', targets: [{ format: this.format, blend }] },
      primitive: { topology: opts.topology ?? 'triangle-list', cullMode: opts.cull ?? 'none' },
      depthStencil: { format: DEPTH, depthWriteEnabled: opts.depthWrite ?? true, depthCompare: opts.depthCompare ?? 'less' },
      multisample: { count: MSAA },
    });
  }

  setSolver(solver: SolverGPU) {
    this.solver = solver;
    this.frameBG = this.device.createBindGroup({
      layout: this.frameLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frameBuf } },
        { binding: 1, resource: solver.velTex.createView() },
        { binding: 2, resource: solver.vortTex.createView() },
        { binding: 3, resource: solver.statTex.createView() },
        { binding: 4, resource: this.sampler },
        { binding: 5, resource: { buffer: solver.flags } },
        { binding: 6, resource: solver.meanTex.createView() },
      ],
    });
    this.needFill = true;
  }

  setMesh(mesh: MeshData | null) {
    this.meshVB?.destroy();
    this.meshIB?.destroy();
    this.meshVB = this.meshIB = null;
    this.meshCount = 0;
    if (!mesh) return;
    const nv = mesh.positions.length / 3;
    const inter = new Float32Array(nv * 6);
    for (let i = 0; i < nv; i++) {
      inter.set(mesh.positions.subarray(3 * i, 3 * i + 3), 6 * i);
      inter.set(mesh.normals.subarray(3 * i, 3 * i + 3), 6 * i + 3);
    }
    this.meshVB = this.device.createBuffer({ size: inter.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.meshVB, 0, inter);
    this.meshIB = this.device.createBuffer({ size: Math.ceil(mesh.indices.byteLength / 4) * 4, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.meshIB, 0, mesh.indices);
    this.meshCount = mesh.indices.length;
  }

  /** Re-seed all tracers (after reset / geometry change). */
  refillParticles() {
    this.needFill = true;
  }

  private ensureTracers(count: number, trail: number) {
    if (this.tracers && this.tracers.count === count && this.tracers.trail === trail) return;
    this.tracers?.state.destroy();
    this.tracers?.hist.destroy();
    const d = this.device;
    const state = d.createBuffer({ size: count * 16, usage: GPUBufferUsage.STORAGE });
    const hist = d.createBuffer({ size: count * trail * 16, usage: GPUBufferUsage.STORAGE });
    this.tracers = {
      count, trail, state, hist,
      computeBG: d.createBindGroup({ layout: this.layouts.partC, entries: [
        { binding: 0, resource: { buffer: this.partU } }, { binding: 1, resource: { buffer: state } }, { binding: 2, resource: { buffer: hist } },
      ] }),
    };
    this.trailHead = 0;
    this.needFill = true;
  }

  private trailU: GPUBuffer | null = null;
  private trailBG: GPUBindGroup | null = null;
  private trailBGFor: TracerBuffers | null = null;

  private ensureStreamlines(seeds: number, points: number) {
    if (this.slBuf && this.slSize[0] === seeds && this.slSize[1] === points) return;
    this.slBuf?.destroy();
    const d = this.device;
    this.slBuf = d.createBuffer({ size: seeds * points * 16, usage: GPUBufferUsage.STORAGE });
    this.slBG = d.createBindGroup({ layout: this.layouts.slC, entries: [{ binding: 0, resource: { buffer: this.slU } }, { binding: 1, resource: { buffer: this.slBuf } }] });
    this.slRenderBG = d.createBindGroup({ layout: this.layouts.stream, entries: [{ binding: 0, resource: { buffer: this.slU } }, { binding: 1, resource: { buffer: this.slBuf } }] });
    this.slSize = [seeds, points];
  }

  private resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (w === this.size[0] && h === this.size[1] && this.msaaTex) return;
    this.canvas.width = w;
    this.canvas.height = h;
    this.size = [w, h];
    this.msaaTex?.destroy();
    this.depthTex?.destroy();
    this.msaaTex = this.device.createTexture({ size: [w, h], sampleCount: MSAA, format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.depthTex = this.device.createTexture({ size: [w, h], sampleCount: MSAA, format: DEPTH, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    if (OFFSCREEN) {
      this.offTex?.destroy();
      this.offBuf?.destroy();
      this.offTex = this.device.createTexture({ size: [w, h], format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
      this.offBuf = this.device.createBuffer({ size: w * h * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      this.offStore?.destroy();
      this.offStore = this.device.createBuffer({ size: w * h * 4, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE });
      if (!this.offPipe) {
        this.offPipe = this.device.createComputePipeline({ layout: 'auto', compute: { entryPoint: 'main', module: this.device.createShaderModule({ code: `
          @group(0) @binding(0) var src: texture_2d<f32>;
          @group(0) @binding(1) var<storage, read_write> dst: array<u32>;
          @compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
            let d = textureDimensions(src);
            if (g.x >= d.x || g.y >= d.y) { return; }
            dst[g.x + g.y * d.x] = pack4x8unorm(textureLoad(src, vec2<i32>(g.xy), 0));
          }` }) } });
      }
      this.offBG = this.device.createBindGroup({ layout: this.offPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: this.offTex.createView() }, { binding: 1, resource: { buffer: this.offStore } }] });
    }
  }

  /** Offscreen debug mode: call after queue.submit() to blit the frame into the 2D canvas. */
  afterSubmit() {
    if (!OFFSCREEN || !this.offCopyPending || !this.offBuf) return;
    this.offCopyPending = false;
    this.offBusy = true;
    const [w, h] = this.size;
    const buf = this.offBuf;
    buf.mapAsync(GPUMapMode.READ).then(() => {
      const src = new Uint8Array(buf.getMappedRange());
      const img = new ImageData(w, h);
      img.data.set(src);
      for (let i = 3; i < img.data.length; i += 4) img.data[i] = 255;
      buf.unmap();
      this.off2D!.putImageData(img, 0, 0);
      this.offBusy = false;
    }).catch((e) => {
      console.warn('offscreen map failed', e);
      this.offBusy = false;
    });
  }
  private offCopyPending = false;

  get pixelSize(): [number, number] {
    return [this.size[0], this.size[1]];
  }

  /** Update tracer compute + draw the frame. `enc` already contains the solver passes. */
  render(enc: GPUCommandEncoder, s: RenderState) {
    if (!this.solver || !this.frameBG) return;
    this.resize();
    const d = this.device;
    const sv = this.solver;
    const { nx, ny, nz } = sv.dims;
    const is2D = !sv.is3D;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    // ---- frame uniform
    const fb = new Float32Array(64);
    fb.set(s.viewProj, 0);
    fb.set(s.invViewProj, 16);
    fb.set([s.eye[0], s.eye[1], s.eye[2], 1], 32);
    fb.set([this.size[0], this.size[1], 1 / this.size[0], 1 / this.size[1]], 36);
    fb.set([nx, ny, nz, is2D ? 1 : 0], 40);
    fb.set([s.flow.U, s.flow.rhoRef, s.flow.Lref, s.time], 44);
    fb.set([s.groundOffset, dpr, 0, 0], 48);
    d.queue.writeBuffer(this.frameBuf, 0, fb);

    // ---- tracers (compute)
    const P = s.particles;
    if (P.enabled && !SKIP.has('particles')) {
      this.ensureTracers(P.count, P.trail);
      const t = this.tracers!;
      const fill = this.needFill;
      if (P.advance && !fill) this.trailHead = (this.trailHead + 1) % t.trail;
      const pu = new ArrayBuffer(80);
      const u32 = new Uint32Array(pu);
      const f32 = new Float32Array(pu);
      u32[0] = t.count; u32[1] = t.trail; u32[2] = this.trailHead; u32[3] = P.emitter === 'rake' ? 1 : 0;
      f32[4] = P.steps; u32[5] = this.frameSeed++; f32[6] = P.maxAge; u32[7] = P.nozzles;
      f32.set([...s.rake.a, 0], 8);
      f32.set([...s.rake.b, 0], 12);
      u32[16] = P.advance ? 1 : 0; u32[17] = fill ? 1 : 0;
      d.queue.writeBuffer(this.partU, 0, pu);
      const cp = enc.beginComputePass({ label: 'particles' });
      cp.setPipeline(this.partPipe);
      cp.setBindGroup(0, this.frameBG);
      cp.setBindGroup(1, t.computeBG);
      cp.dispatchWorkgroups(Math.ceil(t.count / 128));
      cp.end();
      this.needFill = false;
      // trail uniform for drawing
      if (!this.trailU) {
        this.trailU = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      }
      const tu = new ArrayBuffer(32);
      const tu32 = new Uint32Array(tu), tf32 = new Float32Array(tu);
      tu32[0] = t.count; tu32[1] = t.trail; tu32[2] = this.trailHead; tu32[3] = P.colorMode === 'speed' ? 1 : 0;
      tf32[4] = P.width * dpr; tf32[5] = P.alpha;
      d.queue.writeBuffer(this.trailU, 0, tu);
      if (!this.trailBG || this.trailBGFor !== t) {
        this.trailBG = d.createBindGroup({ layout: this.layouts.ribbon, entries: [
          { binding: 0, resource: { buffer: this.trailU } }, { binding: 1, resource: { buffer: t.state } }, { binding: 2, resource: { buffer: t.hist } },
        ] });
        this.trailBGFor = t;
      }
    }
    const SL = s.streamlines;
    if (SL.enabled && !SKIP.has('stream')) {
      this.ensureStreamlines(SL.seeds, SL.points);
      const su = new ArrayBuffer(64);
      const u32 = new Uint32Array(su), f32 = new Float32Array(su);
      u32[0] = SL.seeds; u32[1] = SL.points; u32[2] = SL.useMean ? 1 : 0;
      f32.set([...s.rake.a, 0], 4);
      f32.set([...s.rake.b, 0], 8);
      f32[12] = SL.step;
      f32[13] = SL.width * dpr;
      f32[14] = SL.alpha;
      d.queue.writeBuffer(this.slU, 0, su);
      const cp = enc.beginComputePass({ label: 'streamlines' });
      cp.setPipeline(this.slPipe);
      cp.setBindGroup(0, this.frameBG);
      cp.setBindGroup(1, this.slBG!);
      cp.dispatchWorkgroups(Math.ceil(SL.seeds / 64));
      cp.end();
    }

    // ---- uniforms for drawing
    // mesh
    const mu = new ArrayBuffer(160);
    const mf = new Float32Array(mu), mi = new Uint32Array(mu);
    mf.set(s.mesh.model, 0);
    mf.set(s.mesh.normalMatrix, 16);
    mi[32] = s.mesh.mode === 'cp' ? 1 : 0; mi[33] = s.mesh.cmap; mi[34] = s.mesh.useMean ? 1 : 0; mi[35] = is2D ? 1 : 0;
    mf[36] = s.mesh.cpMin; mf[37] = s.mesh.cpMax; mf[38] = s.mesh.opacity;
    d.queue.writeBuffer(this.meshU, 0, mu);
    // slice
    const su = new ArrayBuffer(48);
    const sf = new Float32Array(su), si = new Uint32Array(su);
    si[0] = s.slice.axis; si[1] = s.slice.field; si[2] = s.slice.cmap;
    si[3] = (s.slice.mean ? 1 : 0) | (s.slice.recirc ? 2 : 0) | (s.slice.qContour ? 4 : 0);
    sf[4] = s.slice.pos; sf[5] = s.slice.vmin; sf[6] = s.slice.vmax; sf[7] = s.slice.opacity; sf[8] = s.slice.qThr;
    d.queue.writeBuffer(this.sliceU, 0, su);
    // volume
    const vu = new ArrayBuffer(32);
    const vf = new Float32Array(vu), vi = new Uint32Array(vu);
    vf[0] = s.volume.thr; vi[1] = s.volume.mode === 'recirc' ? 1 : 0; vi[2] = s.volume.colorBy === 'rotation' ? 1 : 0;
    vi[3] = Math.ceil((Math.hypot(nx, ny, nz) * 1.1) / s.volume.step);
    vf[4] = s.volume.step; vf[5] = s.volume.opacity;
    d.queue.writeBuffer(this.volU, 0, vu);
    // ground
    const zPad = is2D ? Math.max(nx * 0.25, 60) : nz * 0.6;
    d.queue.writeBuffer(this.groundU, 0, new Float32Array([-nx * 0.6, nx * 1.6, is2D ? -zPad : -zPad, is2D ? zPad : nz + zPad, s.ground.moving ? 1 : 0, is2D ? 32 : 16, 0, 0]));
    // overlay segments: tunnel box + rake
    const segs: number[] = [];
    const seg = (a: number[], b: number[], w: number, c: number[]) => segs.push(a[0], a[1], a[2], w, b[0], b[1], b[2], 0, c[0], c[1], c[2], c[3]);
    if (s.box) {
      const zb = is2D ? [0.5, 0.5] : [0, nz];
      const X = [0, nx], Y = [0, ny];
      const bc = [0.45, 0.55, 0.7, 0.55];
      for (const y of Y) for (const z of zb) seg([0, y, z], [nx, y, z], 1.2, bc);
      for (const x of X) for (const z of zb) seg([x, 0, z], [x, ny, z], 1.2, bc);
      if (!is2D) for (const x of X) for (const y of Y) seg([x, y, 0], [x, y, nz], 1.2, bc);
    }
    if (s.rake.visible) {
      const rc = s.rake.active ? [1.0, 0.85, 0.3, 1] : [1.0, 0.75, 0.2, 0.85];
      seg(s.rake.a, s.rake.b, 3, rc);
      const m = [(s.rake.a[0] + s.rake.b[0]) / 2, (s.rake.a[1] + s.rake.b[1]) / 2, (s.rake.a[2] + s.rake.b[2]) / 2];
      // handle: a short thick dash in the middle
      const dir = [s.rake.b[0] - s.rake.a[0], s.rake.b[1] - s.rake.a[1], s.rake.b[2] - s.rake.a[2]];
      const dl = Math.hypot(dir[0], dir[1], dir[2]) || 1;
      const hl = Math.max(dl * 0.04, 2);
      seg([m[0] - (dir[0] / dl) * hl, m[1] - (dir[1] / dl) * hl, m[2] - (dir[2] / dl) * hl], [m[0] + (dir[0] / dl) * hl, m[1] + (dir[1] / dl) * hl, m[2] + (dir[2] / dl) * hl], 12, [1, 0.9, 0.4, 1]);
    }
    this.segCount = segs.length / 12;
    if (this.segCount) d.queue.writeBuffer(this.segBuf, 0, new Float32Array(segs));

    // ---- render pass
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: this.msaaTex!.createView(),
        resolveTarget: OFFSCREEN ? this.offTex!.createView() : this.ctx!.getCurrentTexture().createView(),
        clearValue: { r: 0.043, g: 0.047, b: 0.055, a: 1 },
        loadOp: 'clear',
        storeOp: 'discard',
      }],
      depthStencilAttachment: { view: this.depthTex!.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });
    pass.setBindGroup(0, this.frameBG);
    const overlayDepth: GPUCompareFunction = is2D ? 'always' : 'less';

    // ground
    if (!SKIP.has('ground')) {
    pass.setPipeline(this.pipeline('ground', () => this.basicPipe('ground', GROUND_WGSL, this.layouts.uni, {})));
    pass.setBindGroup(1, this.groundBG);
    pass.draw(6);
    }

    // mesh
    if (!SKIP.has('mesh') && s.mesh.visible && this.meshVB && this.meshCount) {
      const transparent = s.mesh.opacity < 0.999;
      pass.setPipeline(this.pipeline(`mesh${transparent ? 't' : ''}`, () => this.device.createRenderPipeline({
        label: 'mesh',
        layout: this.meshPipeLayout,
        vertex: {
          module: this.module(MESH_WGSL, 'mesh'), entryPoint: 'vs',
          buffers: [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' }] }],
        },
        fragment: {
          module: this.module(MESH_WGSL, 'mesh'), entryPoint: 'fs',
          targets: [{ format: this.format, blend: transparent ? { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } } : undefined }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: DEPTH, depthWriteEnabled: !transparent, depthCompare: 'less' },
        multisample: { count: MSAA },
      })));
      pass.setBindGroup(1, this.meshBG);
      pass.setVertexBuffer(0, this.meshVB);
      pass.setIndexBuffer(this.meshIB!, 'uint32');
      pass.drawIndexed(this.meshCount);
    }

    // slice heatmap
    if (s.slice.visible && !SKIP.has('slice')) {
      const tr = s.slice.opacity < 0.999;
      pass.setPipeline(this.pipeline(`slice${tr ? 't' : ''}`, () => this.basicPipe('slice', SLICE_WGSL, this.layouts.uni, { blend: tr ? 'premul' : 'none', depthWrite: !tr })));
      pass.setBindGroup(1, this.sliceBG);
      pass.draw(6);
    }

    // vortex / recirculation iso-surface
    if (s.volume.visible && !is2D && !SKIP.has('volume')) {
      pass.setPipeline(this.pipeline('volume', () => this.basicPipe('volume', VOLUME_WGSL, this.layouts.uni, { blend: 'premul', depthWrite: true })));
      pass.setBindGroup(1, this.volBG);
      pass.draw(3);
    }

    // streamlines
    if (SL.enabled && this.slRenderBG && !SKIP.has('stream')) {
      pass.setPipeline(this.pipeline(`stream-${overlayDepth}`, () => this.basicPipe('stream', STREAM_WGSL, this.layouts.stream, { blend: 'premul', depthWrite: false, depthCompare: overlayDepth })));
      pass.setBindGroup(1, this.slRenderBG);
      pass.draw((SL.points - 1) * 6, SL.seeds);
    }

    // particle trails
    if (P.enabled && this.tracers && this.trailBG && !SKIP.has('trails')) {
      pass.setPipeline(this.pipeline(`trails-${overlayDepth}`, () => this.basicPipe('trails', TRAILS_WGSL, this.layouts.ribbon, { blend: 'premul', depthWrite: false, depthCompare: overlayDepth })));
      pass.setBindGroup(1, this.trailBG);
      pass.draw((this.tracers.trail - 1) * 6, this.tracers.count);
    }

    // overlay segments
    if (this.segCount && !SKIP.has('segments')) {
      pass.setPipeline(this.pipeline('segments', () => this.basicPipe('segments', SEGMENTS_WGSL, this.layouts.seg, { blend: 'premul', depthWrite: false, depthCompare: 'always' })));
      pass.setBindGroup(1, this.segBG);
      pass.draw(6, this.segCount);
    }
    pass.end();
    if (OFFSCREEN && !this.offBusy && this.offFrame++ % 4 === 0) {
      const [w, h] = this.size;
      const cp = enc.beginComputePass();
      cp.setPipeline(this.offPipe!);
      cp.setBindGroup(0, this.offBG!);
      cp.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
      cp.end();
      enc.copyBufferToBuffer(this.offStore!, 0, this.offBuf!, 0, w * h * 4);
      this.offCopyPending = true;
    }
  }
}
