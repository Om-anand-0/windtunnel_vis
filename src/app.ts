import * as THREE from 'three';
import { coefficients, dragNewtons, Ema, Series, windowStats } from './analysis/aero';
import { FlowMapping, mapFlow } from './analysis/units';
import type { Backend } from './backend/types';
import { CameraRig } from './render/camera';
import { COLORMAPS } from './render/colormaps';
import type { RenderState } from './render/renderState';
import type { SolverStats } from './solver/types';
import { csv, download, stamp, vtk } from './analysis/export';
import { applyPartial, loadSaved, save, shareUrl } from './persist';
import { defaultSettings, FIELDS, GRID_2D, GRID_3D, Quality, Settings } from './state';
import { Capture } from './ui/capture';
import { Hud } from './ui/hud';
import { Panel } from './ui/panel';
import { maybeStartTour } from './ui/tour';
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
  /** standard error of the windowed C_D mean */
  cdSE: number;
  /** averaging window actually covered (convective times) */
  avgSpan: number;
  converged: boolean;
  /** C_D corrected for solid blockage (continuity): C_D·(1−ε)² */
  cdCorrected: number;
}

export interface Probe {
  pos: [number, number, number];
  color: [number, number, number];
  speed: Series;
  cp: Series;
  last: { speed: number; cp: number; cpMean: number; speedMean: number } | null;
}

const PROBE_COLORS: [number, number, number][] = [
  [1, 0.42, 0.42], [0.35, 0.82, 1], [0.7, 0.55, 1], [0.45, 0.9, 0.55], [1, 0.62, 0.25], [1, 0.45, 0.8], [0.9, 0.9, 0.4], [0.6, 0.95, 0.95],
];

/** averaging window for the reported coefficients, in convective times L/U */
const AVG_WINDOW = 4;

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

  readonly cdSeries = new Series(3000);
  readonly cdAvgSeries = new Series(3000);
  readonly clAvgSeries = new Series(3000);
  /** raw post-warm-up samples used for the windowed means */
  readonly raw = { cd: new Series(20000), cl: new Series(20000), cs: new Series(20000) };
  private cdAvg = new Ema(0.05);
  private clAvg = new Ema(0.05);
  private csAvg = new Ema(0.05);
  private lastStats: SolverStats | null = null;
  private rakeActive = false;
  /** set once the user moves the Q threshold slider */
  isoThrUser = false;
  private startCam: number[] | null = null;

  constructor(readonly backend: Backend) {
    this.rig = new CameraRig(backend.canvas, backend.kind === 'webgpu');
    const q = new URLSearchParams(location.search);
    // a shared link reproduces a view exactly (defaults + URL); otherwise restore the last session
    const shared = [...q.keys()].some((k) => k.startsWith('s.') || k === 'cam');
    const saved = shared ? {} : loadSaved();
    applyPartial(this.s, saved);
    const fresh = !shared && Object.keys(saved).length === 0;
    if (q.get('mode') === '2d' || q.get('mode') === '3d') this.s.mode = q.get('mode') as '2d' | '3d';
    // WebGL2 fallback has no 3D solver
    const forced2D = !backend.supports3D && this.s.mode === '3d';
    if (!backend.supports3D) this.s.mode = '2d';
    if (q.get('vehicle')) this.s.vehicle = q.get('vehicle')!;
    if (q.get('quality')) {
      const v = q.get('quality') as Quality;
      this.s.quality2D = v;
      this.s.quality3D = v;
    }
    if (fresh || forced2D) this.applyModeDefaults();
    this.startCam = q.get('cam')?.split(',').map(Number).filter((x) => isFinite(x)) ?? null;
    window.addEventListener('beforeunload', () => save(this.s));
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
    if (this.startCam?.length === 6) this.rig.setView(this.startCam.slice(0, 3), this.startCam.slice(3));
    else setTimeout(maybeStartTour, 800);
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
    for (const r of Object.values(this.raw)) r.clear();
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
    this.raw.cd.push(tConv, c.cd);
    this.raw.cl.push(tConv, c.cl);
    this.raw.cs.push(tConv, c.cs);
    this.cdSeries.push(tConv, c.cd);
    this.cdAvgSeries.push(tConv, this.cdAvg.value);
    this.clAvgSeries.push(tConv, this.clAvg.value);
  };
  private lastSampleStep = 0;
  private cdInst = NaN;
  rhoRef = 1;

  readout(): AeroReadout {
    const w = (s: Series) => windowStats(s.t, s.v, AVG_WINDOW);
    const sd = w(this.raw.cd), sl = w(this.raw.cl), ss = w(this.raw.cs);
    const pick = (st: { mean: number }, ema: Ema) => (isFinite(st.mean) ? st.mean : ema.value);
    const cd = pick(sd, this.cdAvg);
    const speed = this.s.speedKmh / 3.6;
    const blockage = this.vinfo.frontal / (this.is3D ? this.dims.ny * this.dims.nz : this.dims.ny);
    const converged = sd.span >= AVG_WINDOW * 0.75 && isFinite(sd.se) && sd.se < Math.max(0.02 * Math.abs(sd.mean), 0.005);
    return {
      cd, cl: pick(sl, this.clAvg), cs: pick(ss, this.csAvg), cdInst: this.cdInst,
      cdSE: sd.se, avgSpan: sd.span, converged, cdCorrected: cd * (1 - blockage) ** 2,
      frontalM2: this.frontalM2,
      dragN: dragNewtons(cd, speed, this.frontalM2),
      flow: this.flow,
      stepsPerFrame: this.paused ? 0 : Math.round(this.stepsAuto * this.s.simSpeed),
      fps: this.fps,
      mlups: (this.backend.cells * (this.fixedSpf || Math.round(this.stepsAuto * this.s.simSpeed)) * this.fps) / 1e6,
      convTime: (this.backend.stepCount * this.cur.U) / this.Lcells,
      maxMach: this.lastStats ? this.lastStats.maxU * Math.sqrt(3) : 0,
      blockage,
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

  // ---------------------------------------------------------------- export

  private metaLines(): string[] {
    const d = this.dims, f = this.flow, r = this.readout();
    return [
      `windtunnel export ${new Date().toISOString()}`,
      `vehicle=${this.presetInfo?.name ?? 'uploaded model'} length_m=${this.s.lengthM} speed_kmh=${this.s.speedKmh} yaw_deg=${this.s.yaw} pitch_deg=${this.s.pitch} ride_mm=${this.s.rideMm}`,
      `solver=${this.is3D ? 'D3Q19' : 'D2Q9'} grid=${d.nx}x${d.ny}x${d.nz} L_cells=${this.Lcells.toFixed(1)} U_lattice=${this.cur.U.toFixed(4)} tau=${f.tau.toFixed(5)} ground=${this.s.ground} Cs=${this.s.lesCs}`,
      `Re_real=${f.reReal.toExponential(3)} Re_sim=${f.reSim.toFixed(0)} dx_m=${f.dx.toExponential(4)} dt_s=${f.dt.toExponential(4)} frontal_m2=${this.frontalM2.toFixed(4)} blockage=${r.blockage.toFixed(4)}`,
      `Cd=${r.cd.toFixed(4)} +- ${r.cdSE.toFixed(4)} Cl=${r.cl.toFixed(4)} Cs=${r.cs.toFixed(4)} converged=${r.converged}`,
      't = convective time t*U/L',
    ];
  }

  exportCSV() {
    const r = this.raw;
    // probe traces are merged in as extra columns (nearest sample in time) — one file, one download
    const nearest = (ser: Series, t: number) => {
      const ts = ser.t;
      if (!ts.length) return NaN;
      let lo = 0, hi = ts.length - 1;
      while (hi - lo > 1) { const m = (lo + hi) >> 1; if (ts[m] < t) lo = m; else hi = m; }
      const k = Math.abs(ts[lo] - t) < Math.abs(ts[hi] - t) ? lo : hi;
      return Math.abs(ts[k] - t) < 0.05 ? ser.v[k] : NaN;
    };
    const header = ['t', 'cd', 'cl', 'cs'];
    const meta = this.metaLines();
    this.probes.forEach((p, k) => {
      header.push(`P${k + 1}_speed_over_U`, `P${k + 1}_cp`);
      meta.push(`P${k + 1} at (${this.probeMetres(p).map((x) => x.toFixed(3)).join(', ')}) m from the vehicle centre`);
    });
    const rows = r.cd.t.map((t, i) => {
      const row: number[] = [t, r.cd.v[i], r.cl.v[i] ?? NaN, r.cs.v[i] ?? NaN];
      for (const p of this.probes) row.push(nearest(p.speed, t), nearest(p.cp, t));
      return row;
    });
    download(csv(header, rows, meta), `windtunnel-${stamp()}.csv`);
  }

  async exportVTK() {
    toast('Reading the flow field back from the GPU…', 'info', 2500);
    const f = await this.backend.exportFields(1_000_000);
    const blob = vtk(f, { dx: this.flow.dx, U: Math.max(this.cur.U, 1e-6), L: this.Lcells, rhoRef: this.lastStats?.rhoRef ?? 1, title: this.metaLines().slice(1, 3).join(' | ') });
    download(blob, `windtunnel-field-${f.dims.join('x')}-${stamp()}.vtk`);
    toast(`Saved ${f.dims.join('×')} points${f.stride > 1 ? ` (every ${f.stride}th cell)` : ''} — open in ParaView`, 'info');
  }

  /** Link that reproduces the current settings and camera. */
  shareLink(): string {
    const c = this.rig.camera.position, t = this.rig.controls.target;
    return shareUrl(this.s, { pos: [c.x, c.y, c.z], target: [t.x, t.y, t.z] });
  }

  // ---------------------------------------------------------------- probes

  readonly probes: Probe[] = [];
  placingProbe = false;

  /** Put a probe where the click ray meets the slice plane (or the centre plane). */
  private placeProbe(x: number, y: number, w: number, h: number) {
    const { nx, ny, nz } = this.dims;
    const ray = this.rig.ray(x, y, w, h);
    const axis = this.is3D && this.s.sliceOn ? this.s.sliceAxis : 2;
    const dimsArr = [nx, ny, nz];
    const pos = this.is3D ? (this.s.sliceOn ? this.s.slicePos * dimsArr[axis] : nz / 2) : 0.5;
    const n = new THREE.Vector3(axis === 0 ? 1 : 0, axis === 1 ? 1 : 0, axis === 2 ? 1 : 0);
    const plane = new THREE.Plane(n, -pos);
    const hit = new THREE.Vector3();
    this.placingProbe = false;
    this.backend.canvas.style.cursor = '';
    if (!ray.intersectPlane(plane, hit) || hit.x < 0 || hit.y < 0 || hit.x > nx || hit.y > ny || (this.is3D && (hit.z < 0 || hit.z > nz))) {
      toast('Click inside the tunnel on the slice plane to place a probe', 'warn');
      this.panel.sync();
      return;
    }
    if (!this.is3D) hit.z = 0.5;
    const color = PROBE_COLORS[this.probes.length % PROBE_COLORS.length];
    this.probes.push({ pos: [hit.x, hit.y, hit.z], color, speed: new Series(3000), cp: new Series(3000), last: null });
    this.panel.sync();
  }

  startProbePlacement() {
    if (this.probes.length >= 8) {
      toast('Up to 8 probes — clear some first', 'warn');
      return;
    }
    this.placingProbe = true;
    this.backend.canvas.style.cursor = 'crosshair';
  }

  clearProbes() {
    this.probes.length = 0;
    this.panel.sync();
  }

  /** Probe position in metres from the vehicle centre (for display). */
  probeMetres(p: Probe): [number, number, number] {
    const dx = this.flow.dx;
    const c = new THREE.Vector3().setFromMatrixPosition(this.meshMatrix);
    return [(p.pos[0] - c.x) * dx, p.pos[1] * dx, this.is3D ? (p.pos[2] - c.z) * dx : 0];
  }

  private updateProbes() {
    const data = this.backend.probeData();
    const U = Math.max(this.cur.U, 1e-6);
    const rhoRef = this.lastStats?.rhoRef ?? 1;
    const t = (this.backend.stepCount * this.cur.U) / this.Lcells;
    data.forEach((d, i) => {
      const p = this.probes[i];
      if (!p) return;
      const sp = Math.hypot(d.inst[0], d.inst[1], d.inst[2]) / U;
      const cp = (2 * (d.inst[3] - rhoRef)) / (3 * U * U);
      const cpMean = (2 * (d.mean[3] - rhoRef)) / (3 * U * U);
      const spMean = Math.hypot(d.mean[0], d.mean[1], d.mean[2]) / U;
      p.last = { speed: sp, cp, cpMean, speedMean: spMean };
      const lt = p.speed.t.length ? p.speed.t[p.speed.t.length - 1] : -1;
      if (t > lt) {
        p.speed.push(t, sp);
        p.cp.push(t, cp);
      }
    });
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
      if (this.placingProbe && e.button === 0) {
        const r = c.getBoundingClientRect();
        this.placeProbe(e.clientX - r.left, e.clientY - r.top, r.width, r.height);
        e.stopImmediatePropagation();
        return;
      }
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
      probes: this.probes.map((p) => ({ pos: p.pos, color: p.color })),
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

    if (this.probes.length) this.updateProbes();
    if (this.frames % 6 === 0) this.hud.update();
    if (this.frames % 180 === 0) save(this.s);
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
