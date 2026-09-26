import * as THREE from 'three';
import { coefficients, dragNewtons, Ema, Series } from './analysis/aero';
import { FlowMapping, mapFlow } from './analysis/units';
import type { Backend } from './backend/types';
import { CameraRig } from './render/camera';
import { COLORMAPS } from './render/colormaps';
import type { RenderState } from './render/renderState';
import type { SolverStats } from './solver/types';
import { defaultSettings, FIELDS, GRID_2D, GRID_3D, Quality, Settings } from './state';
import { Capture } from './ui/capture';
import { Hud } from './ui/hud';
import { Panel } from './ui/panel';
import { ValidationPanel } from './ui/validationPanel';
import { toast } from './ui/widgets';
import { LoadedModel } from './voxelize/loaders';
import { frontalAreaNormalized, MeshData, normalizeMesh, placementMatrix } from './voxelize/mesh';
import { buildPreset, PRESETS } from './voxelize/presets';
import type { VoxelInfo } from './voxelize/VoxelizerGPU';

const V_MAX_KMH = 300;
const QUALITY_ORDER: Quality[] = ['low', 'medium', 'high', 'ultra'];

export interface AeroReadout {
  cd: number;
  cl: number;
  cs: number;
  cdInst: number;
  frontalM2: number;
  dragN: number;
  flow: FlowMapping;
  stepsPerFrame: number;
  fps: number;
  mlups: number;
  convTime: number;
  maxMach: number;
  /** frontal area / tunnel cross-section */
  blockage: number;
}

export class App {
  readonly s: Settings = defaultSettings();
  readonly rig: CameraRig;
  private ready = false;
  panel!: Panel;
  hud!: Hud;
  validation!: ValidationPanel;
  capture!: Capture;

  mesh: MeshData | null = null;
  private upload: LoadedModel | null = null;
  private meshMatrix = new THREE.Matrix4();
  private vinfo: VoxelInfo = { frontal: 1, solidCells: 0, min: [0, 0, 0], max: [0, 0, 0] };
  private frontalM2 = 2;
  private Lcells = 100;

  flow!: FlowMapping;
  private cur = { U: 0.05, nu: 0.01 };
  private stab = { uScale: 1, nuScale: 1, events: 0 };
  private stepsAuto = 8;
  private stepRequest = 0;
  /** debug/benchmark: ?spf=N forces N steps per frame */
  private fixedSpf = parseInt(new URLSearchParams(location.search).get('spf') ?? '0') || 0;
  private frames = 0;
  private fps = 60;
  private fpsAcc = { t: performance.now(), n: 0 };
  private framesSinceReset = 0;
  private groundOffset = 0;
  private busy = false;
  private revoxTimer = 0;
  private perfWatch = { t0: 0, frames: 0, active: false };

  readonly cdSeries = new Series(900);
  readonly cdAvgSeries = new Series(900);
  readonly clAvgSeries = new Series(900);
  private cdAvg = new Ema(0.05);
  private clAvg = new Ema(0.05);
  private csAvg = new Ema(0.05);
  private lastStats: SolverStats | null = null;
  private rakeActive = false;
  /** set once the user moves the Q threshold slider */
  isoThrUser = false;

  constructor(readonly backend: Backend) {
    this.rig = new CameraRig(backend.canvas, backend.kind === 'webgpu');
    const q = new URLSearchParams(location.search);
    if (q.get('mode') === '2d' || q.get('mode') === '3d') this.s.mode = q.get('mode') as '2d' | '3d';
    // WebGL2 fallback has no 3D solver
    if (!backend.supports3D) this.s.mode = '2d';
    if (q.get('vehicle')) this.s.vehicle = q.get('vehicle')!;
    if (q.get('quality')) {
      const v = q.get('quality') as Quality;
      this.s.quality2D = v;
      this.s.quality3D = v;
    }
    this.applyModeDefaults();
    // any setting can be overridden from the URL, e.g. ?s.field=3&s.isoOn=1 (shareable views)
    const rec = this.s as unknown as Record<string, unknown>;
    for (const [k, v] of q) {
      if (!k.startsWith('s.')) continue;
      const key = k.slice(2);
      if (!(key in rec)) continue;
      const cur = rec[key];
      rec[key] = typeof cur === 'number' ? parseFloat(v) : typeof cur === 'boolean' ? v === '1' || v === 'true' : v;
      if (key === 'isoThr') this.isoThrUser = true;
    }
  }

  async init() {
    this.panel = new Panel(this, document.getElementById('panel')!);
    this.hud = new Hud(this, document.getElementById('hud')!, document.getElementById('topbar')!);
    this.validation = new ValidationPanel(this);
    this.capture = new Capture(this);
    this.setupRakeDrag();
    await this.rebuildSolver();
    requestAnimationFrame(this.loop);
  }

  // ---------------------------------------------------------------- configuration

  get dims() {
    return this.s.mode === '3d' ? GRID_3D[this.s.quality3D] : GRID_2D[this.s.quality2D];
  }

  get is3D() {
    return this.s.mode === '3d';
  }

  get presetInfo() {
    return PRESETS.find((p) => p.id === this.s.vehicle);
  }

  private applyModeDefaults() {
    const p = this.presetInfo;
    if (p?.placement === 'center') this.s.ground = 'freeslip';
    if (this.is3D) {
      this.s.bodyOn = true;
      this.s.sliceAxis = 2;
      this.s.slicePos = 0.5;
      this.s.camera = 'free';
      this.s.particleCount = 150000;
      this.s.trail = 20;
      this.s.streamOn = true;
    } else {
      this.s.bodyOn = false;
      this.s.sliceAxis = 2;
      this.s.camera = 'side';
      this.s.particleCount = 100000;
      this.s.trail = 16;
      this.s.streamOn = false;
    }
  }

  /** Fits the requested grid into device limits, stepping the quality down if necessary. */
  private fitQuality(): boolean {
    let changed = false;
    while (true) {
      if (this.backend.fits(this.dims)) return changed;
      const key = this.is3D ? 'quality3D' : 'quality2D';
      const i = QUALITY_ORDER.indexOf(this.s[key]);
      if (i <= 0) return changed;
      this.s[key] = QUALITY_ORDER[i - 1];
      changed = true;
    }
  }

  async rebuildSolver() {
    this.busy = true;
    if (this.fitQuality()) toast(`Grid reduced to "${this.is3D ? this.s.quality3D : this.s.quality2D}" to fit this GPU's buffer limits`, 'warn');
    const d = this.dims;
    this.updateFlow();
    this.cur = { U: this.flow.U, nu: this.flow.nu };
    await this.backend.createSolver(d, {
      u: this.cur.U, nu: this.cur.nu, cs: this.s.lesCs, ground: this.s.ground, sides: 'freeslip',
      spongeNu: 0.04, spongeStart: 0.84, emaAlpha: 0.02, tauWall: 0.53, spongeIn: 0.03,
    });
    this.ready = true;
    this.rig.setDomain(d);
    this.rig.goTo(this.s.camera, false);
    this.stepsAuto = this.is3D ? 2 : 8;
    this.stab = { uScale: 1, nuScale: 1, events: 0 };
    await this.loadVehicle();
    this.resetFlow();
    this.panel?.sync();
    this.busy = false;
    if (this.is3D) this.perfWatch = { t0: performance.now(), frames: 0, active: true };
  }

  /** Build or fetch the mesh for the current vehicle setting and voxelize it. */
  async loadVehicle() {
    const p = this.presetInfo;
    if (this.s.vehicle === 'upload' && this.upload) {
      this.mesh = normalizeMesh(this.upload.raw, { up: this.s.upAxis, flip: this.s.flip });
    } else if (p) {
      this.mesh = buildPreset(p.id);
      this.s.lengthM = p.lengthM;
    }
    this.backend.setMesh(this.mesh);
    await this.revoxelize();
  }

  setUpload(m: LoadedModel) {
    this.upload = m;
    this.s.vehicle = 'upload';
    this.s.upAxis = m.up;
    this.s.flip = false;
    this.s.ground = 'moving';
    this.s.lengthM = 4.5;
  }

  private placement() {
    const p = this.presetInfo;
    const center = p?.placement === 'center';
    const d = this.dims;
    const dxReal = this.s.lengthM / (d.nx / 3);
    return {
      mode: center ? ('center' as const) : ('ground' as const),
      lengthFrac: 1 / 3,
      diamFrac: this.is3D ? 0.2 : 0.1,
      xFrac: center ? 0.25 : 0.34,
      yawDeg: this.is3D ? this.s.yaw : 0,
      pitchDeg: this.s.pitch,
      rideCells: center ? 0 : this.s.rideMm / 1000 / dxReal,
      spanwise: p?.spanwise,
    };
  }

  async revoxelize() {
    if (!this.mesh) return;
    const d = this.dims;
    const pl = this.placement();
    const { matrix, scale } = placementMatrix(this.mesh, d, pl);
    this.meshMatrix.copy(matrix);
    const center = pl.mode === 'center';
    // reference length: vehicle length or body diameter
    this.Lcells = center ? pl.diamFrac * d.ny : scale;
    this.vinfo = await this.backend.voxelize(this.mesh, matrix.elements);
    // vortex cores are a few cells across, so Q·L²/U² there scales like L²: pick a matching default
    if (!this.isoThrUser) this.s.isoThr = Math.round(0.015 * this.Lcells * this.Lcells);
    const aN = frontalAreaNormalized(this.mesh, pl.yawDeg, pl.pitchDeg);
    if (center) {
      const p = this.presetInfo!;
      const Dm = p.lengthM;
      this.frontalM2 = p.spanwise ? Dm * 1 : (Math.PI * Dm * Dm) / 4;
    } else {
      this.frontalM2 = aN * this.s.lengthM * this.s.lengthM;
    }
    this.backend.clearStats();
    this.framesSinceReset = 0;
    this.updateFlow();
    this.resetCoefficients();
    this.backend.refillParticles();
  }

  /** Debounced re-voxelization for interactive sliders (yaw, pitch, ride height). */
  requestRevoxelize() {
    clearTimeout(this.revoxTimer);
    this.revoxTimer = window.setTimeout(() => this.revoxelize(), 60);
  }

  updateFlow() {
    const p = this.presetInfo;
    const center = p?.placement === 'center';
    const L = this.Lcells;
    this.flow = mapFlow({
      speed: this.s.speedKmh / 3.6,
      lengthM: this.s.lengthM,
      nuAir: this.s.nuAir * 1e-5,
      Lcells: L,
      vMax: V_MAX_KMH / 3.6,
      nuMin: this.is3D ? 0.0006 : 0.0003,
      reOverride: this.s.reOverrideOn ? this.s.reOverride : center ? this.defaultCenterRe() : null,
      uScale: this.stab.uScale,
      nuScale: this.stab.nuScale,
    });
  }

  /** Validation bodies map the speed slider onto a laminar/transitional Re range. */
  private defaultCenterRe() {
    return 20 + (this.s.speedKmh / V_MAX_KMH) * (this.is3D ? 800 : 1500);
  }

  resetFlow() {
    this.backend.reset();
    this.cur = { U: this.flow.U, nu: this.flow.nu };
    this.framesSinceReset = 0;
    this.resetCoefficients();
  }

  private resetCoefficients() {
    this.cdSeries.clear();
    this.cdAvgSeries.clear();
    this.clAvgSeries.clear();
    this.cdAvg.reset();
    this.clAvg.reset();
    this.csAvg.reset();
  }

  step() {
    this.stepRequest = Math.max(1, Math.round(this.stepsAuto));
  }

  // ---------------------------------------------------------------- stability

  private lastStabFrame = -1000;

  private handleInstability(reason: string) {
    this.lastStabFrame = this.frames;
    this.stab.events++;
    if (this.stab.uScale > 0.55) this.stab.uScale *= 0.85;
    else this.stab.nuScale *= 1.6;
    this.stab.nuScale *= 1.25;
    this.updateFlow();
    this.resetFlow();
    toast(`Solver instability (${reason}). Auto-stabilised: lattice speed ×${this.stab.uScale.toFixed(2)}, viscosity ×${this.stab.nuScale.toFixed(2)} → Re_sim ${Math.round(this.flow.reSim)}`, 'warn', 6500);
  }

  // ---------------------------------------------------------------- statistics

  private onStats = (st: SolverStats) => {
    if (st.step === 0) return;
    // samples read back asynchronously may predate the last reset: drop them
    if (st.step > this.backend.stepCount) return;
    this.lastStats = st;
    if ((st.unstable || st.maxU * Math.sqrt(3) > 0.6) && this.frames - this.lastStabFrame < 30) return;
    if (st.unstable) {
      this.handleInstability(isFinite(st.fx) ? `Ma = ${(st.maxU * Math.sqrt(3)).toFixed(2)}` : 'NaN');
      return;
    }
    if (st.maxU * Math.sqrt(3) > 0.6) {
      this.handleInstability(`local Mach ${(st.maxU * Math.sqrt(3)).toFixed(2)} > 0.6`);
      return;
    }
    const U = this.cur.U;
    const area = Math.max(this.vinfo.frontal, 1);
    const c = coefficients(st.fx, st.fy, st.fz, U, area, st.rhoRef);
    const tConv = (st.step * U) / this.Lcells;
    // averaging window ~1.5 convective times
    const dt = this.lastSampleStep ? st.step - this.lastSampleStep : 0;
    this.lastSampleStep = st.step;
    const k = dt > 0 ? 1 - Math.exp(-(dt * U) / (1.5 * this.Lcells)) : 0.05;
    this.cdAvg.k = this.clAvg.k = this.csAvg.k = Math.min(Math.max(k, 0.002), 1);
    this.cdInst = c.cd;
    // the first convective time is the start-up transient: plot it, but keep it out of the average
    if (tConv < 1) {
      this.cdSeries.push(tConv, c.cd);
      return;
    }
    this.cdAvg.push(c.cd);
    this.clAvg.push(c.cl);
    this.csAvg.push(c.cs);
    this.cdSeries.push(tConv, c.cd);
    this.cdAvgSeries.push(tConv, this.cdAvg.value);
    this.clAvgSeries.push(tConv, this.clAvg.value);
  };
  private lastSampleStep = 0;
  private cdInst = NaN;
  rhoRef = 1;

  readout(): AeroReadout {
    const cd = this.cdAvg.value;
    const speed = this.s.speedKmh / 3.6;
    return {
      cd, cl: this.clAvg.value, cs: this.csAvg.value, cdInst: this.cdInst,
      frontalM2: this.frontalM2,
      dragN: dragNewtons(cd, speed, this.frontalM2),
      flow: this.flow,
      stepsPerFrame: this.paused ? 0 : Math.round(this.stepsAuto * this.s.simSpeed),
      fps: this.fps,
      mlups: (this.backend.cells * (this.fixedSpf || Math.round(this.stepsAuto * this.s.simSpeed)) * this.fps) / 1e6,
      convTime: (this.backend.stepCount * this.cur.U) / this.Lcells,
      maxMach: this.lastStats ? this.lastStats.maxU * Math.sqrt(3) : 0,
      blockage: this.vinfo.frontal / (this.is3D ? this.dims.ny * this.dims.nz : this.dims.ny),
    };
  }

  get paused() {
    return this.s.paused;
  }

  get lengthCells() {
    return this.Lcells;
  }

  get currentU() {
    return this.cur.U;
  }

  get stabilityEvents() {
    return this.stab.events;
  }

  // ---------------------------------------------------------------- rake

  rakeEndpoints(): { a: [number, number, number]; b: [number, number, number] } {
    const { nx, ny, nz } = this.dims;
    const s = this.s;
    const x = s.rakeX * nx;
    const z = this.is3D ? s.rakeZ * nz : 0.5;
    const y = s.rakeY * ny;
    if (s.rakeOrient === 'horizontal' && this.is3D) {
      const half = (s.rakeSpan * nz) / 2;
      return { a: [x, y, Math.max(1, z - half)], b: [x, y, Math.min(nz - 1, z + half)] };
    }
    const half = (s.rakeSpan * ny) / 2;
    return { a: [x, Math.max(0.5, y - half), z], b: [x, Math.min(ny - 1, y + half), z] };
  }

  private setupRakeDrag() {
    const c = this.backend.canvas;
    let dragging = false;
    let plane: THREE.Plane | null = null;
    let offset = new THREE.Vector3();
    const handlePos = () => {
      const { a, b } = this.rakeEndpoints();
      return new THREE.Vector3((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2);
    };
    const rakeVisible = () => this.s.streamOn || (this.s.particlesOn && this.s.emitter === 'rake');
    c.addEventListener('pointerdown', (e) => {
      if (!rakeVisible() || e.button !== 0) return;
      const r = c.getBoundingClientRect();
      const hp = handlePos();
      const sp = this.rig.project(hp, r.width, r.height);
      if (Math.hypot(sp.x - (e.clientX - r.left), sp.y - (e.clientY - r.top)) > 18) return;
      dragging = true;
      this.rakeActive = true;
      this.rig.controls.enabled = false;
      c.setPointerCapture(e.pointerId);
      const n = new THREE.Vector3();
      this.rig.camera.getWorldDirection(n);
      plane = new THREE.Plane().setFromNormalAndCoplanarPoint(this.is3D ? n.negate() : new THREE.Vector3(0, 0, 1), hp);
      const ray = this.rig.ray(e.clientX - r.left, e.clientY - r.top, r.width, r.height);
      const hit = new THREE.Vector3();
      if (ray.intersectPlane(plane, hit)) offset = hp.clone().sub(hit);
      e.stopImmediatePropagation();
    }, { capture: true });
    c.addEventListener('pointermove', (e) => {
      const r = c.getBoundingClientRect();
      if (!dragging) {
        if (rakeVisible()) {
          const sp = this.rig.project(handlePos(), r.width, r.height);
          const near = Math.hypot(sp.x - (e.clientX - r.left), sp.y - (e.clientY - r.top)) < 18;
          c.style.cursor = near ? 'grab' : '';
          this.rakeActive = near;
        }
        return;
      }
      const ray = this.rig.ray(e.clientX - r.left, e.clientY - r.top, r.width, r.height);
      const hit = new THREE.Vector3();
      if (!plane || !ray.intersectPlane(plane, hit)) return;
      hit.add(offset);
      const { nx, ny, nz } = this.dims;
      this.s.rakeX = Math.min(Math.max(hit.x / nx, 0.01), 0.95);
      this.s.rakeY = Math.min(Math.max(hit.y / ny, 0.01), 0.95);
      if (this.is3D) this.s.rakeZ = Math.min(Math.max(hit.z / nz, 0.02), 0.98);
      this.panel.sync();
      if (this.s.emitter === 'rake') this.backend.refillParticles();
    });
    const end = (e: PointerEvent) => {
      if (!dragging) return;
      dragging = false;
      this.rakeActive = false;
      this.rig.controls.enabled = true;
      c.releasePointerCapture(e.pointerId);
    };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
  }

  // ---------------------------------------------------------------- frame

  private buildRenderState(steps: number): RenderState {
    const s = this.s;
    const d = this.dims;
    const r = this.backend.pixelSize();
    this.rig.update(r[0], r[1]);
    const m = this.rig.matrices();
    const field = FIELDS[s.field];
    const range = s.autoRange ? (this.is3D && field.range3D ? field.range3D : field.range) : [s.vmin, s.vmax];
    const axis = this.is3D ? s.sliceAxis : 2;
    const dimsArr = [d.nx, d.ny, d.nz];
    const slicePos = this.is3D ? Math.min(Math.max(s.slicePos, 0.002), 0.998) * dimsArr[axis] : 0.5;
    const rk = this.rakeEndpoints();
    const nm = new THREE.Matrix4().copy(this.meshMatrix).invert().transpose();
    return {
      viewProj: m.viewProj,
      invViewProj: m.invViewProj,
      eye: m.eye,
      time: performance.now() / 1000,
      groundOffset: this.groundOffset,
      flow: { U: this.cur.U, rhoRef: this.lastStats?.rhoRef ?? 1, Lref: this.Lcells },
      box: true,
      ground: { moving: s.ground === 'moving' },
      mesh: {
        visible: s.bodyOn,
        model: new Float32Array(this.meshMatrix.elements),
        normalMatrix: new Float32Array(nm.elements),
        mode: s.surface,
        cmap: COLORMAPS.indexOf('turbo'),
        useMean: s.surfaceMean,
        cpMin: -1.5,
        cpMax: 1.0,
        opacity: this.is3D ? 1 : 0.92,
      },
      slice: {
        visible: s.sliceOn,
        axis: axis as 0 | 1 | 2,
        field: s.field,
        cmap: COLORMAPS.indexOf(s.cmap),
        pos: slicePos,
        vmin: range[0],
        vmax: range[1],
        opacity: s.sliceOpacity,
        mean: s.sliceMean,
        recirc: s.recirc,
        qContour: s.qContour,
        qThr: s.isoThr,
      },
      volume: {
        visible: s.isoOn && this.is3D,
        mode: s.isoMode,
        colorBy: s.isoColor,
        thr: s.isoThr,
        step: 0.6,
        opacity: 1,
      },
      particles: {
        enabled: s.particlesOn,
        count: s.particleCount,
        trail: s.trail,
        emitter: s.emitter,
        colorMode: s.smokeColor,
        width: this.is3D ? 1.1 : 1.2,
        alpha: s.emitter === 'rake' ? (this.is3D ? 0.35 : 0.38) : this.is3D ? 0.25 : 0.3,
        advance: steps > 0,
        steps,
        maxAge: Math.min(Math.max((1.2 * d.nx) / Math.max(this.cur.U * Math.max(steps, 1), 1e-3), 60), 6000),
        nozzles: this.is3D ? 18 : 30,
      },
      streamlines: {
        enabled: s.streamOn,
        seeds: s.seeds,
        points: this.is3D ? 700 : 1400,
        step: 0.7,
        useMean: s.streamMean,
        width: 1.6,
        alpha: 0.9,
      },
      rake: { a: rk.a, b: rk.b, visible: s.streamOn || (s.particlesOn && s.emitter === 'rake'), active: this.rakeActive },
    };
  }

  private loop = (now: number) => {
    requestAnimationFrame(this.loop);
    if (this.busy || !this.ready) return;
    // never queue more than two frames of GPU work (keeps latency low and the step tuner honest)
    if (this.backend.throttled()) return;
    this.fpsAcc.n++;
    if (now - this.fpsAcc.t > 500) {
      this.fps = (this.fpsAcc.n * 1000) / (now - this.fpsAcc.t);
      this.fpsAcc = { t: now, n: 0 };
    }
    this.frames++;

    // smooth parameter ramps (no solver shocks on slider moves)
    const k = 0.06;
    this.cur.U += (this.flow.U - this.cur.U) * k;
    // start-up damping: extra viscosity for the first ~convective time damps the acoustic
    // transient of the impulsive start, then decays away
    const tConv = (this.backend.stepCount * this.cur.U) / this.Lcells;
    const nuTarget = Math.min(this.flow.nu * (1 + 12 * Math.exp(-tConv / 0.25)), 0.08);
    this.cur.nu = Math.exp(Math.log(this.cur.nu) + (Math.log(nuTarget) - Math.log(this.cur.nu)) * 0.25);

    let steps = 0;
    if (!this.s.paused) steps = this.fixedSpf || Math.max(1, Math.round(this.stepsAuto * this.s.simSpeed));
    else if (this.stepRequest) { steps = this.stepRequest; this.stepRequest = 0; }

    const avgTime = 3 * this.Lcells / Math.max(this.cur.U, 1e-4);
    let ema = 1 - Math.exp(-Math.max(steps, 1) / avgTime);
    this.framesSinceReset++;
    ema = Math.max(ema, 1 / this.framesSinceReset);
    this.backend.setParams({ u: this.cur.U, nu: this.cur.nu, cs: this.s.lesCs, ground: this.s.ground, emaAlpha: steps > 0 ? ema : 0 });
    this.groundOffset += this.s.ground === 'moving' ? this.cur.U * steps : 0;

    const rs = this.buildRenderState(steps);
    this.backend.frame(steps, rs, this.onStats);
    this.capture.afterFrame();

    // adapt steps/frame to the GPU time budget
    // GPU-time says how much headroom there is; the measured frame rate has the final word, so a
    // pessimistic timer can't throttle a fast GPU and an optimistic one can't drop frames
    if (!this.s.paused && this.frames > 10) {
      const budget = (1000 / this.s.targetFps) * 0.8;
      const ratio = budget / Math.max(this.backend.gpuMs, 0.1);
      let f = Math.min(Math.max(ratio, 0.85), 1.06);
      const fpsOk = this.fps >= this.s.targetFps * 0.93;
      if (fpsOk && f < 1) f = 1;
      if (this.fps < this.s.targetFps * 0.8) f = Math.min(f, 0.95);
      this.stepsAuto = Math.min(Math.max(this.stepsAuto * f + (ratio > 1.2 && fpsOk ? 0.2 : 0), 1), this.is3D ? 60 : 400);
    }

    // slow-device guard for 3D
    if (this.perfWatch.active) {
      this.perfWatch.frames++;
      const el = now - this.perfWatch.t0;
      if (el > 4000) {
        this.perfWatch.active = false;
        const fps = (this.perfWatch.frames * 1000) / el;
        if (fps < 12 && this.stepsAuto <= 1.5) this.degrade3D(fps);
      }
    }

    if (this.frames % 6 === 0) this.hud.update();
  };

  private degrade3D(fps: number) {
    const i = QUALITY_ORDER.indexOf(this.s.quality3D);
    if (i > 0) {
      this.s.quality3D = QUALITY_ORDER[i - 1];
      toast(`3D running at ${fps.toFixed(0)} fps — lowering grid to "${this.s.quality3D}"`, 'warn');
      this.rebuildSolver();
    } else {
      toast(`3D is too slow on this device (${fps.toFixed(0)} fps). Falling back to the 2D side-profile solver.`, 'warn', 8000);
      this.setMode('2d');
    }
  }

  async setMode(mode: '2d' | '3d') {
    if (this.s.mode === mode) return;
    if (mode === '3d' && !this.backend.supports3D) {
      toast('3D mode needs WebGPU (compute shaders). This browser is running the WebGL2 fallback — 2D only.', 'warn', 7000);
      this.panel.sync();
      return;
    }
    this.s.mode = mode;
    this.applyModeDefaults();
    await this.rebuildSolver();
  }

  async setVehicle(id: string) {
    this.s.vehicle = id;
    const p = this.presetInfo;
    if (p) {
      this.s.lengthM = p.lengthM;
      this.s.ground = p.placement === 'center' ? 'freeslip' : 'moving';
      if (p.placement === 'center') { this.s.yaw = 0; this.s.pitch = 0; }
    }
    this.busy = true;
    await this.loadVehicle();
    this.updateFlow();
    this.resetFlow();
    this.busy = false;
    this.panel.sync();
  }
}
