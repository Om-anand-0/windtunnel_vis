import type { App } from '../app';
import { formatRe } from '../analysis/units';
import { COLORMAPS } from '../render/colormaps';
import { FIELDS, GRID_2D, GRID_3D, QUALITIES, Quality } from '../state';
import { clearSaved } from '../persist';
import { loadModelFile } from '../voxelize/loaders';
import { PRESETS } from '../voxelize/presets';
import { applyButtonTips, BUTTON_TIPS, installTooltips, PANEL_TIPS } from './tips';
import { buttonRow, Control, h, note, section, segmented, select, slider, toast, toggle } from './widgets';

/** The left-hand control panel. Every control writes into app.s and triggers the right side effect. */
export class Panel {
  private ctl: Record<string, Control<any> & { setHint?(s: string): void }> = {};
  private pauseBtn!: HTMLButtonElement;
  private placeBtn!: HTMLButtonElement;

  constructor(private app: App, root: HTMLElement) {
    const s = app.s;
    root.innerHTML = '';

    // ---------------------------------------------------------- simulation
    const sim = section(root, 'Simulation');
    this.ctl.mode = segmented(sim, {
      label: 'Solver',
      options: [
        { value: '2d', label: '2D  D2Q9', title: 'Centre-plane side profile, high resolution' },
        { value: '3d', label: '3D  D3Q19', title: 'Full 3D (WebGPU)' },
      ],
      value: s.mode,
      onChange: (v) => app.setMode(v as '2d' | '3d'),
    });
    const qopts = (m: '2d' | '3d') =>
      QUALITIES.map((q) => {
        const g = m === '3d' ? GRID_3D[q] : GRID_2D[q];
        return { value: q, label: `${q[0].toUpperCase() + q.slice(1)} · ${g.nx}×${g.ny}${g.nz > 1 ? '×' + g.nz : ''}` };
      });
    this.ctl.quality = select<Quality>(sim, {
      label: 'Grid',
      options: qopts(s.mode),
      value: s.mode === '3d' ? s.quality3D : s.quality2D,
      onChange: (v) => {
        if (app.is3D) s.quality3D = v; else s.quality2D = v;
        app.rebuildSolver();
      },
    });
    (this.ctl.quality as any).refreshOptions = () => (this.ctl.quality as any).setOptions(qopts(s.mode));
    this.ctl.precision = segmented(sim, {
      label: 'Storage', options: [{ value: 'f16', label: 'FP16 (fast)' }, { value: 'f32', label: 'FP32' }], value: s.precision,
      onChange: (v) => { s.precision = v as 'f16' | 'f32'; app.rebuildSolver(); },
    });
    const [pb] = buttonRow(sim, [
      { label: 'Pause', onClick: () => { s.paused = !s.paused; this.sync(); }, title: 'Space' },
      { label: 'Step', onClick: () => app.step(), title: 'Advance one frame of steps (.)' },
      { label: 'Reset', onClick: () => app.resetFlow(), title: 'Re-initialise the flow (R)' },
    ]);
    this.pauseBtn = pb;
    this.ctl.simSpeed = slider(sim, {
      label: 'Sim speed', min: 0.1, max: 3, step: 0.05, value: s.simSpeed,
      format: (v) => `${v.toFixed(2)}×`, onInput: (v) => (s.simSpeed = v),
      hint: 'Multiplier on the auto-tuned steps per frame',
    });
    this.ctl.targetFps = segmented(sim, {
      label: 'Target fps', options: [{ value: 30, label: '30' }, { value: 45, label: '45' }, { value: 60, label: '60' }], value: s.targetFps,
      onChange: (v) => (s.targetFps = v as number),
    });

    // ---------------------------------------------------------- vehicle
    const veh = section(root, 'Vehicle');
    const vopts = () => [
      ...PRESETS.map((p) => ({ value: p.id, label: p.name })),
      ...(s.vehicle === 'upload' ? [{ value: 'upload', label: 'Uploaded model' }] : []),
    ];
    this.ctl.vehicle = select<string>(veh, { label: 'Model', options: vopts(), value: s.vehicle, onChange: (v) => app.setVehicle(v) });
    (this.ctl.vehicle as any).refreshOptions = () => (this.ctl.vehicle as any).setOptions(vopts());
    const file = h('input', { type: 'file', accept: '.glb,.gltf,.obj,.stl', style: 'display:none' });
    veh.append(file);
    const [upBtn] = buttonRow(veh, [{ label: 'Upload .glb / .obj / .stl…', onClick: () => file.click(), cls: 'wide' }]);
    const handleFile = async (f: File) => {
      upBtn.textContent = 'Loading…';
      try {
        const m = await loadModelFile(f);
        app.setUpload(m);
        await app.setVehicle('upload');
        toast(`Loaded ${m.name} — ${m.triangles.toLocaleString()} triangles`, 'info');
      } catch (e) {
        toast(String((e as Error).message ?? e), 'error', 7000);
      }
      upBtn.textContent = 'Upload .glb / .obj / .stl…';
      this.sync();
    };
    file.addEventListener('change', () => file.files?.[0] && handleFile(file.files[0]));
    document.body.addEventListener('dragover', (e) => e.preventDefault());
    document.body.addEventListener('drop', (e) => {
      e.preventDefault();
      const f = e.dataTransfer?.files?.[0];
      if (f) handleFile(f);
    });
    this.ctl.upAxis = segmented(veh, {
      label: 'Up axis', options: [{ value: 'y', label: 'Y-up' }, { value: 'z', label: 'Z-up' }], value: s.upAxis,
      onChange: (v) => { s.upAxis = v as 'y' | 'z'; app.loadVehicle(); },
    });
    const [flipBtn] = buttonRow(veh, [{ label: 'Flip driving direction', onClick: () => { s.flip = !s.flip; app.loadVehicle(); }, cls: 'wide' }]);
    this.ctl.flip = { el: flipBtn.parentElement!, get: () => s.flip, set: () => {}, setEnabled: (on) => (flipBtn.disabled = !on), setVisible: (on) => (flipBtn.parentElement!.style.display = on ? '' : 'none') };
    this.ctl.lengthM = slider(veh, {
      label: 'Real length', min: 0.05, max: 25, step: 0.05, value: s.lengthM, log: true,
      format: (v) => `${v < 1 ? v.toFixed(2) : v.toFixed(1)} m`,
      onInput: (v) => { s.lengthM = v; app.revoxelize(); },
    });
    this.ctl.yaw = slider(veh, {
      label: 'Yaw (crosswind)', min: -30, max: 30, step: 0.5, value: s.yaw, format: (v) => `${v.toFixed(1)}°`,
      onInput: (v) => { s.yaw = v; app.requestRevoxelize(); },
    });
    this.ctl.pitch = slider(veh, {
      label: 'Pitch', min: -6, max: 6, step: 0.1, value: s.pitch, format: (v) => `${v.toFixed(1)}°`,
      onInput: (v) => { s.pitch = v; app.requestRevoxelize(); },
    });
    this.ctl.ride = slider(veh, {
      label: 'Ride height offset', min: 0, max: 300, step: 5, value: s.rideMm, format: (v) => `${v.toFixed(0)} mm`,
      onInput: (v) => { s.rideMm = v; app.requestRevoxelize(); },
    });

    // ---------------------------------------------------------- flow
    const flow = section(root, 'Flow');
    this.ctl.speed = slider(flow, {
      label: 'Wind speed', min: 10, max: 300, step: 1, value: s.speedKmh, format: (v) => `${v.toFixed(0)} km/h`,
      onInput: (v) => { s.speedKmh = v; app.flowChanged(); this.updateHints(); },
    });
    this.ctl.nuAir = slider(flow, {
      label: 'Air viscosity ν', min: 0.5, max: 5, step: 0.05, value: s.nuAir, format: (v) => `${v.toFixed(2)}·10⁻⁵ m²/s`,
      onInput: (v) => { s.nuAir = v; app.flowChanged(); this.updateHints(); },
    });
    this.ctl.reOn = toggle(flow, {
      label: 'Override simulated Re', value: s.reOverrideOn,
      onChange: (v) => { s.reOverrideOn = v; app.flowChanged(); this.updateHints(); this.sync(); },
    });
    this.ctl.re = slider(flow, {
      label: 'Re (sim)', min: 20, max: 200000, step: 1, value: s.reOverride, log: true, format: (v) => formatRe(v),
      onInput: (v) => { s.reOverride = v; app.flowChanged(); this.updateHints(); },
    });
    this.ctl.ground = select<string>(flow, {
      label: 'Ground',
      options: [
        { value: 'moving', label: 'Moving belt (rolling road)' },
        { value: 'noslip', label: 'Fixed floor (no-slip)' },
        { value: 'freeslip', label: 'Free-slip (symmetry)' },
      ],
      value: s.ground,
      onChange: (v) => { s.ground = v as typeof s.ground; app.flowChanged(); },
    });
    this.ctl.interp = toggle(flow, {
      label: 'Interpolated walls', value: s.interpWalls,
      onChange: (v) => { s.interpWalls = v; app.flowChanged(); },
    });
    this.ctl.wheels = toggle(flow, {
      label: 'Rotating wheels', value: s.rotatingWheels,
      onChange: (v) => { s.rotatingWheels = v; app.revoxelize(); },
    });
    this.ctl.les = slider(flow, {
      label: 'Smagorinsky Cₛ', min: 0, max: 0.3, step: 0.01, value: s.lesCs, format: (v) => v.toFixed(2),
      onInput: (v) => (s.lesCs = v), hint: 'LES sub-grid model strength (0 = DNS)',
    });

    // ---------------------------------------------------------- heatmap
    const heat = section(root, 'Heatmap slice');
    this.ctl.sliceOn = toggle(heat, { label: 'Show slice', value: s.sliceOn, onChange: (v) => (s.sliceOn = v) });
    this.ctl.field = select<number>(heat, {
      label: 'Field', options: FIELDS.map((f) => ({ value: f.id, label: f.label })), value: s.field,
      onChange: (v) => { s.field = v; s.cmap = FIELDS[v].cmap; this.sync(); },
    });
    this.ctl.cmap = select<string>(heat, {
      label: 'Colormap', options: COLORMAPS.map((c) => ({ value: c, label: c })), value: s.cmap,
      onChange: (v) => (s.cmap = v as typeof s.cmap),
    });
    this.ctl.sliceMean = toggle(heat, { label: 'Time-averaged', value: s.sliceMean, onChange: (v) => (s.sliceMean = v) });
    this.ctl.sliceAxis = segmented(heat, {
      label: 'Plane', options: [{ value: 2, label: 'Side XY' }, { value: 1, label: 'Top XZ' }, { value: 0, label: 'Cross YZ' }], value: s.sliceAxis,
      onChange: (v) => { s.sliceAxis = v as 0 | 1 | 2; s.slicePos = v === 2 ? 0.5 : v === 1 ? 0.12 : 0.55; this.sync(); },
    });
    this.ctl.slicePos = slider(heat, {
      label: 'Slice position', min: 0, max: 1, step: 0.002, value: s.slicePos, format: (v) => `${(v * 100).toFixed(1)} %`,
      onInput: (v) => (s.slicePos = v),
    });
    this.ctl.opacity = slider(heat, {
      label: 'Opacity', min: 0.1, max: 1, step: 0.01, value: s.sliceOpacity, format: (v) => v.toFixed(2), onInput: (v) => (s.sliceOpacity = v),
    });
    this.ctl.autoRange = toggle(heat, { label: 'Default range', value: s.autoRange, onChange: (v) => { s.autoRange = v; this.sync(); } });
    this.ctl.vmin = slider(heat, { label: 'Min', min: -300, max: 300, step: 0.01, value: s.vmin, format: (v) => v.toFixed(2), onInput: (v) => (s.vmin = v) });
    this.ctl.vmax = slider(heat, { label: 'Max', min: -300, max: 300, step: 0.01, value: s.vmax, format: (v) => v.toFixed(2), onInput: (v) => (s.vmax = v) });
    this.ctl.recirc = toggle(heat, { label: 'Recirculation contour (ūₓ = 0)', value: s.recirc, onChange: (v) => (s.recirc = v) });
    this.ctl.qContour = toggle(heat, { label: 'Vortex-core contour (Q)', value: s.qContour, onChange: (v) => (s.qContour = v) });

    // ---------------------------------------------------------- body
    const body = section(root, 'Body surface');
    this.ctl.bodyOn = toggle(body, { label: 'Show 3D body', value: s.bodyOn, onChange: (v) => (s.bodyOn = v) });
    this.ctl.surface = segmented(body, {
      label: 'Shading', options: [{ value: 'cp', label: 'Cp heatmap' }, { value: 'lit', label: 'Studio' }], value: s.surface,
      onChange: (v) => (s.surface = v as 'cp' | 'lit'),
    });
    this.ctl.surfaceMean = toggle(body, { label: 'Time-averaged Cp', value: s.surfaceMean, onChange: (v) => (s.surfaceMean = v) });

    // ---------------------------------------------------------- particles / streamlines
    const tr = section(root, 'Smoke & streamlines');
    this.ctl.particlesOn = toggle(tr, { label: 'Smoke particles', value: s.particlesOn, onChange: (v) => (s.particlesOn = v) });
    this.ctl.emitter = segmented(tr, {
      label: 'Emitter', options: [{ value: 'rake', label: 'Smoke rake' }, { value: 'inlet', label: 'Whole inlet' }], value: s.emitter,
      onChange: (v) => { s.emitter = v as 'rake' | 'inlet'; app.backend.refillParticles(); },
    });
    this.ctl.smokeColor = segmented(tr, {
      label: 'Colour', options: [{ value: 'smoke', label: 'Smoke' }, { value: 'speed', label: 'Speed' }], value: s.smokeColor,
      onChange: (v) => (s.smokeColor = v as 'smoke' | 'speed'),
    });
    this.ctl.count = slider(tr, {
      label: 'Particles', min: 5000, max: 400000, step: 1000, value: s.particleCount, log: true,
      format: (v) => `${Math.round(v / 1000)}k`, onInput: (v) => (s.particleCount = Math.round(v / 1000) * 1000),
    });
    this.ctl.trail = slider(tr, { label: 'Trail length', min: 2, max: 48, step: 1, value: s.trail, format: (v) => `${v} frames`, onInput: (v) => (s.trail = v) });
    this.ctl.streamOn = toggle(tr, { label: 'Streamlines', value: s.streamOn, onChange: (v) => (s.streamOn = v) });
    this.ctl.seeds = slider(tr, { label: 'Seeds', min: 4, max: 160, step: 1, value: s.seeds, format: (v) => String(v), onInput: (v) => (s.seeds = v) });
    this.ctl.streamMean = toggle(tr, { label: 'Use time-averaged field', value: s.streamMean, onChange: (v) => (s.streamMean = v) });
    this.ctl.rakeOrient = segmented(tr, {
      label: 'Rake', options: [{ value: 'vertical', label: 'Vertical' }, { value: 'horizontal', label: 'Horizontal' }], value: s.rakeOrient,
      onChange: (v) => { s.rakeOrient = v as 'vertical' | 'horizontal'; app.backend.refillParticles(); },
    });
    this.ctl.rakeSpan = slider(tr, { label: 'Rake span', min: 0.05, max: 1, step: 0.01, value: s.rakeSpan, format: (v) => `${(v * 100).toFixed(0)} %`, onInput: (v) => { s.rakeSpan = v; app.backend.refillParticles(); } });
    note(tr, 'Drag the yellow handle in the viewport to move the rake.');

    // ---------------------------------------------------------- probes
    const pr = section(root, 'Probes', false);
    const [placeBtn] = buttonRow(pr, [
      { label: 'Place probe', onClick: () => { app.startProbePlacement(); this.sync(); } },
      { label: 'Clear probes', onClick: () => app.clearProbes() },
    ]);
    this.placeBtn = placeBtn;
    note(pr, 'Click “Place probe”, then click on the heatmap slice in the viewport. Up to 8 probes; values and a Cp trace appear in the right panel and in the CSV export.');

    // ---------------------------------------------------------- wake
    const wake = section(root, 'Wake & vortices', false);
    this.ctl.isoOn = toggle(wake, { label: 'Iso-surface (3D)', value: s.isoOn, onChange: (v) => (s.isoOn = v) });
    this.ctl.isoMode = segmented(wake, {
      label: 'Field', options: [{ value: 'q', label: 'Q-criterion' }, { value: 'recirc', label: 'Separation (ūₓ<0)' }], value: s.isoMode,
      onChange: (v) => (s.isoMode = v as 'q' | 'recirc'),
    });
    this.ctl.isoThr = slider(wake, {
      label: 'Q threshold', min: 5, max: 20000, step: 1, value: s.isoThr, log: true, format: (v) => `${v.toFixed(0)} U²/L²`,
      onInput: (v) => { s.isoThr = v; app.isoThrUser = true; }, hint: 'Also used for the vortex-core contour on the slice',
    });
    this.ctl.isoColor = segmented(wake, {
      label: 'Colour', options: [{ value: 'speed', label: 'Speed' }, { value: 'rotation', label: 'Rotation (ωₓ)' }], value: s.isoColor,
      onChange: (v) => (s.isoColor = v as 'speed' | 'rotation'),
    });

    // ---------------------------------------------------------- camera / export
    const cam = section(root, 'Camera & export');
    this.ctl.camera = segmented(cam, {
      options: [{ value: 'side', label: 'Side' }, { value: 'top', label: 'Top' }, { value: 'front', label: 'Front' }, { value: 'free', label: 'Orbit' }],
      value: s.camera,
      onChange: (v) => { s.camera = v as typeof s.camera; app.rig.goTo(s.camera); },
    });
    const [, recBtn] = buttonRow(cam, [
      { label: 'Screenshot', onClick: () => app.capture.screenshot(), title: 'P' },
      { label: 'Record video', onClick: () => { app.capture.toggleRecording(); recBtn.textContent = app.capture.recording ? '■ Stop recording' : 'Record video'; recBtn.classList.toggle('rec', app.capture.recording); } },
    ]);
    const [, gifBtn] = buttonRow(cam, [
      { label: 'Export CSV', onClick: () => app.exportCSV() },
      { label: 'Record GIF', onClick: () => app.capture.recordGif((on) => { gifBtn.textContent = on ? 'Recording GIF…' : 'Record GIF'; gifBtn.classList.toggle('rec', on); }) },
    ]);
    buttonRow(cam, [{ label: 'Export flow field (.vtk)', onClick: () => app.exportVTK(), cls: 'wide' }]);
    buttonRow(cam, [
      {
        label: 'Copy link', onClick: async () => {
          const url = app.shareLink();
          try { await navigator.clipboard.writeText(url); toast('Link to this view copied to the clipboard', 'info'); }
          catch { window.prompt('Copy this link:', url); }
        },
      },
      { label: 'Reset settings', onClick: () => { clearSaved(); location.href = location.pathname; } },
    ]);
    buttonRow(cam, [{ label: 'Studies: compare & sweep…', onClick: () => app.studies.open(), cls: 'wide' }]);
    buttonRow(cam, [{ label: 'Validation suite…', onClick: () => app.validation.open(), cls: 'wide accent' }]);

    for (const [k, tip] of Object.entries(PANEL_TIPS)) if (this.ctl[k]) this.ctl[k].el.dataset.tip = tip;
    applyButtonTips(root);
    installTooltips();

    this.updateHints();
    this.sync();
    window.addEventListener('keydown', (e) => {
      if ((e.target as HTMLElement).tagName === 'INPUT' && (e.target as HTMLInputElement).type !== 'range') return;
      if (e.code === 'Space') { s.paused = !s.paused; this.sync(); e.preventDefault(); }
      else if (e.key === 'r' || e.key === 'R') app.resetFlow();
      else if (e.key === '.') app.step();
      else if (e.key === 'p' || e.key === 'P') app.capture.screenshot();
      else if (e.key === '1') { s.camera = 'side'; app.rig.goTo('side'); this.sync(); }
      else if (e.key === '2') { s.camera = 'top'; app.rig.goTo('top'); this.sync(); }
      else if (e.key === '3') { s.camera = 'front'; app.rig.goTo('front'); this.sync(); }
      else if (e.key === '4') { s.camera = 'free'; app.rig.goTo('free'); this.sync(); }
    });
  }

  updateHints() {
    const f = this.app.flow;
    if (!f) return;
    this.ctl.speed.setHint?.(`Re real ${formatRe(f.reReal)} · Re sim ${formatRe(f.reSim)} · Ma ${f.mach.toFixed(3)}`);
    this.ctl.re.setHint?.(`τ = ${f.tau.toFixed(4)}  (ν_lattice = ${f.nu.toExponential(2)})`);
  }

  /** Push app state back into the controls (after programmatic changes). */
  sync() {
    const s = this.app.s;
    const is3D = s.mode === '3d';
    const c = this.ctl;
    c.mode.set(s.mode);
    (c.quality as any).refreshOptions();
    c.quality.set(is3D ? s.quality3D : s.quality2D);
    c.precision.set(s.precision);
    c.precision.setVisible(this.app.backend.kind === 'webgpu');
    (c.vehicle as any).refreshOptions();
    c.vehicle.set(s.vehicle);
    this.pauseBtn.textContent = s.paused ? '▶ Run' : '❚❚ Pause';
    this.placeBtn.textContent = this.app.placingProbe ? 'Click in the view…' : `Place probe (${this.app.probes.length}/8)`;
    this.placeBtn.classList.toggle('accent', this.app.placingProbe);
    this.pauseBtn.classList.toggle('accent', s.paused);
    this.pauseBtn.dataset.tip = BUTTON_TIPS.find(([k]) => this.pauseBtn.textContent!.startsWith(k))?.[1] ?? '';
    c.simSpeed.set(s.simSpeed);
    c.lengthM.set(s.lengthM);
    c.yaw.set(s.yaw);
    c.pitch.set(s.pitch);
    c.ride.set(s.rideMm);
    const isUpload = s.vehicle === 'upload';
    c.upAxis.setVisible(isUpload);
    c.upAxis.set(s.upAxis);
    c.flip.setVisible(isUpload);
    const center = this.app.presetInfo?.placement === 'center';
    c.yaw.setEnabled(!center && is3D);
    c.yaw.setHint?.(is3D ? '' : 'Crosswind needs the 3D solver');
    c.pitch.setEnabled(!center);
    c.ride.setEnabled(!center);
    c.speed.set(s.speedKmh);
    c.reOn.set(s.reOverrideOn);
    c.re.setEnabled(s.reOverrideOn);
    c.ground.set(s.ground);
    c.wheels.set(s.rotatingWheels);
    c.interp.set(s.interpWalls);
    c.wheels.setEnabled(is3D && !!this.app.mesh?.wheels?.length);
    c.les.set(s.lesCs);
    c.sliceOn.set(s.sliceOn);
    c.field.set(s.field);
    c.cmap.set(s.cmap);
    c.sliceMean.set(s.sliceMean);
    c.sliceAxis.setVisible(is3D);
    c.sliceAxis.set(s.sliceAxis);
    c.slicePos.setVisible(is3D);
    c.slicePos.set(s.slicePos);
    c.autoRange.set(s.autoRange);
    const fd = FIELDS[s.field];
    const r = is3D && fd.range3D ? fd.range3D : fd.range;
    if (s.autoRange) { s.vmin = r[0]; s.vmax = r[1]; }
    c.vmin.set(s.vmin);
    c.vmax.set(s.vmax);
    c.vmin.setVisible(!s.autoRange);
    c.vmax.setVisible(!s.autoRange);
    c.bodyOn.set(s.bodyOn);
    c.surface.set(s.surface);
    c.count.set(s.particleCount);
    c.trail.set(s.trail);
    c.particlesOn.set(s.particlesOn);
    c.emitter.set(s.emitter);
    c.streamOn.set(s.streamOn);
    c.rakeOrient.setVisible(is3D);
    c.isoOn.setEnabled(is3D);
    c.isoThr.set(s.isoThr);
    c.camera.set(s.camera);
    this.updateHints();
  }
}
