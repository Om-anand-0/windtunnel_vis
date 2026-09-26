/** Hover help for every control and readout. Keys match Panel.ctl / Hud.el names. */

export const PANEL_TIPS: Record<string, string> = {
  mode: '2D runs a high-resolution D2Q9 solver on the vehicle’s centre-plane slice (fast, no wheels). 3D runs the full D3Q19 solver around the whole vehicle (needs WebGPU).',
  quality: 'Grid resolution. More cells resolve finer flow detail and allow a higher Reynolds number, but each step costs proportionally more GPU time. If a grid does not fit this GPU’s buffer limits it is reduced automatically.',
  simSpeed: 'Multiplier on the automatically tuned number of solver steps per frame. Above 1× the flow develops faster but the frame rate drops; below 1× gives slow motion.',
  targetFps: 'Frame-rate the step tuner aims for. Lower targets leave more GPU time per frame for the solver, so the flow evolves faster on screen.',
  vehicle: 'Pick a procedural preset or your uploaded model. Sphere and cylinder are validation bodies placed mid-tunnel instead of on the floor.',
  upAxis: 'Which axis of the uploaded file points up. glTF/OBJ are usually Y-up, STL/CAD exports are often Z-up.',
  flip: 'Turn the uploaded model 180° if its nose points downstream. The wind blows from left (inlet) to right (outlet).',
  lengthM: 'Real-world length of the vehicle. Sets the physical scale: real Reynolds number, cell size Δx and the drag force in newtons.',
  yaw: 'Rotates the vehicle about the vertical axis — equivalent to a crosswind at this angle. Produces side force C_S. 3D only.',
  pitch: 'Nose-up (+) / nose-down (−) rotation. Changes ground effect and lift.',
  ride: 'Raises the vehicle above the floor. More underbody clearance usually reduces ground-effect downforce.',
  speed: 'Free-stream wind speed. Sets the real Reynolds number; the simulated Re scales with it (the grid cannot reach real-car Re, see README), and the flow visibly speeds up.',
  nuAir: 'Kinematic viscosity of the fluid (air ≈ 1.5·10⁻⁵ m²/s). Higher viscosity → lower Reynolds number → thicker, calmer wake.',
  reOn: 'Ignore the speed/viscosity mapping and set the simulated Reynolds number directly.',
  re: 'Simulated Reynolds number U·L/ν on the lattice. Low (≲200): laminar, steady or regular vortex shedding. High: turbulent wake handled by the LES model.',
  ground: 'Moving belt = rolling-road tunnel (floor moves with the air, no floor boundary layer). Fixed floor grows a boundary layer like a static tunnel. Free-slip is a frictionless symmetry plane.',
  les: 'Strength of the Smagorinsky sub-grid turbulence model. It adds eddy viscosity where the flow is under-resolved and keeps high-Re runs stable. 0 = pure DNS (only safe at low Re).',
  sliceOn: 'Show the colour-mapped field on a plane through the tunnel.',
  field: 'Velocity magnitude; streamwise velocity (blue = reversed flow); pressure coefficient Cp (red = high pressure); vorticity (rotation); turbulence intensity (velocity fluctuations); Q-criterion (positive = vortex cores).',
  cmap: 'Colour scale. Turbo/viridis/inferno are perceptual sequential maps; coolwarm is diverging (good for signed fields like Cp, vorticity, uₓ).',
  sliceMean: 'Show the time-averaged field (running average over ~3 flow-through times) instead of the instantaneous snapshot.',
  sliceAxis: 'Orientation of the slice: side (XY), top (XZ, horizontal) or cross-flow (YZ, looks down the wake — best for trailing vortices).',
  slicePos: 'Position of the slice along its normal axis, as a fraction of the tunnel.',
  opacity: 'Slice transparency — lower it to see the body and iso-surfaces behind the plane.',
  autoRange: 'Use the recommended colour range for the selected field. Turn off to set min/max manually.',
  vmin: 'Value mapped to the low end of the colour scale.',
  vmax: 'Value mapped to the high end of the colour scale.',
  recirc: 'Outline where the time-averaged streamwise velocity is zero and tint the reversed-flow region — the separation / recirculation bubble.',
  qContour: 'Outline vortex cores on the slice (Q above the threshold set under “Wake & vortices”).',
  bodyOn: 'Draw the 3D vehicle mesh. In 2D it hides the slice behind it, so it is off by default there.',
  surface: 'Cp heatmap paints the surface pressure coefficient on the body (red = stagnation/high pressure, blue = suction). Studio shows plain shaded geometry.',
  surfaceMean: 'Use the time-averaged pressure for the surface map — much less noisy than the instantaneous value in a turbulent wake.',
  particlesOn: 'GPU smoke tracers advected through the live velocity field.',
  emitter: 'Smoke rake releases streaks from nozzles along the yellow rake (like a real smoke tunnel). Whole inlet seeds particles uniformly across the tunnel entrance.',
  smokeColor: 'White smoke, or colour each particle by its local speed.',
  count: 'Number of tracer particles. More = denser smoke; costs GPU time mainly in drawing the trails.',
  trail: 'How many frames of history each particle draws as a fading trail.',
  streamOn: 'Streamlines integrated through the velocity field from seeds on the rake, coloured by speed. Unlike smoke they show the instantaneous flow direction everywhere along the line.',
  seeds: 'Number of streamlines seeded along the rake.',
  streamMean: 'Integrate streamlines through the time-averaged field — shows the mean flow topology (e.g. the closed recirculation bubble) instead of a snapshot.',
  rakeOrient: 'Vertical rake (spans height) or horizontal rake (spans width). Drag the yellow handle in the viewport to move it.',
  rakeSpan: 'Length of the rake as a fraction of the tunnel height (or width).',
  isoOn: '3D surface drawn where the chosen field crosses its threshold — visualises the wake structure. 3D only.',
  isoMode: 'Q-criterion: vortex tubes (A-pillar, C-pillar and trailing vortices). Separation: the region of time-averaged reversed flow behind the body.',
  isoThr: 'Q-criterion threshold in units of U²/L². Lower shows more (and noisier) structures, higher isolates only the strongest vortex cores. The default adapts to the grid.',
  isoColor: 'Colour the iso-surface by local speed, or by rotation sense (red/blue = opposite-sign streamwise vorticity — pairs of counter-rotating vortices).',
  camera: 'Camera presets (keys 1–4). Orbit = free 3/4 view; drag to orbit, wheel to zoom, right-drag to pan.',
};

/** Tooltips for buttons, keyed by their label text (prefix match). */
export const BUTTON_TIPS: [string, string][] = [
  ['Pause', 'Pause / resume the solver (Space). Rendering keeps running so you can orbit a frozen flow.'],
  ['▶ Run', 'Resume the solver (Space).'],
  ['❚❚ Pause', 'Pause the solver (Space). Rendering keeps running so you can orbit a frozen flow.'],
  ['Step', 'While paused, advance one frame’s worth of solver steps (.)'],
  ['Reset', 'Restart the flow from a uniform free stream (R). Averages and the C_D history are cleared.'],
  ['Upload', 'Load your own model (.glb, .gltf, .obj, .stl) — or drag-and-drop a file anywhere. It is auto-centred, scaled to ⅓ of the tunnel and voxelized on the GPU.'],
  ['Flip', 'Turn the uploaded model 180° about the vertical axis.'],
  ['Screenshot', 'Save a PNG of the viewport with the legend and aero numbers burned in (P).'],
  ['Record', 'Record the viewport to a WebM video; click again to stop and download.'],
  ['■ Stop', 'Stop recording and download the video.'],
  ['Validation', 'Run the physics checks: free-stream uniformity, cylinder vortex shedding (Strouhal number) and sphere drag against published data.'],
];

export const HUD_TIPS: Record<string, string> = {
  backend: 'Compute/graphics API in use. Hover shows the GPU adapter — on a laptop make sure it names the discrete GPU.',
  grid: 'Lattice type, grid dimensions and number of cells.',
  fps: 'Rendered frames per second.',
  spf: 'Solver time steps executed per frame, tuned automatically to hold the target frame rate.',
  mlups: 'Million lattice-cell updates per second — the standard LBM throughput metric.',
  tconv: 'Simulated time in convective units (vehicle lengths travelled by the free stream). Read C_D after ≳3.',
  status: 'Solver state. “Stabilised” means the watchdog caught an instability and raised viscosity / lowered the lattice speed.',
  cd: 'Drag coefficient C_D = F_x / (½ρU²A), time-averaged (the start-up transient is excluded). Computed by momentum exchange on the body surface.',
  cl: 'Lift coefficient C_L = F_y / (½ρU²A). Negative = downforce.',
  cs: 'Side-force coefficient from crosswind (yaw). 3D only.',
  drag: 'Estimated real drag force ½·ρ_air·V²·C_D·A at the chosen wind speed.',
  down: 'Estimated real downforce (−lift) in newtons.',
  area: 'Real frontal area A, rasterised from the mesh as seen from the front.',
  block: 'Share of the tunnel cross-section blocked by the body. Above ~10–15 % the walls speed the flow up and inflate C_D.',
  reReal: 'Reynolds number of the real vehicle at this speed (V·L/ν).',
  reSim: 'Reynolds number actually simulated on the grid (U·L/ν in lattice units).',
  mach: 'Lattice Mach number at the inlet and the current maximum. LBM is weakly compressible; keep the maximum well below ~0.4 for accuracy.',
  tau: 'BGK relaxation time τ = 3ν + ½. Values close to 0.5 mean very low viscosity (high Re) and rely on the LES model for stability.',
  U: 'Inlet velocity in lattice units (cells per time step).',
  dxdt: 'Physical size of one cell and duration of one time step.',
};

/** One floating tooltip for the whole page, driven by [data-tip] attributes. */
export function installTooltips() {
  if (document.getElementById('tip')) return;
  const tip = document.createElement('div');
  tip.id = 'tip';
  document.body.append(tip);
  let timer = 0;
  let current: HTMLElement | null = null;
  const hide = () => {
    clearTimeout(timer);
    current = null;
    tip.classList.remove('show');
  };
  const show = (el: HTMLElement) => {
    tip.textContent = el.dataset.tip ?? '';
    tip.classList.add('show');
    const r = el.getBoundingClientRect();
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    const vw = window.innerWidth, vh = window.innerHeight;
    let x: number, y: number;
    if (r.top < 50) {
      // top bar: below the item
      x = Math.min(Math.max(r.left, 8), vw - tw - 8);
      y = r.bottom + 8;
    } else if (r.right + tw + 14 < vw && r.left < vw / 2) {
      x = r.right + 12;
      y = r.top + r.height / 2 - th / 2;
    } else {
      x = r.left - tw - 12;
      y = r.top + r.height / 2 - th / 2;
    }
    tip.style.left = `${Math.max(8, x)}px`;
    tip.style.top = `${Math.min(Math.max(8, y), vh - th - 8)}px`;
  };
  document.addEventListener('mouseover', (e) => {
    const el = (e.target as HTMLElement).closest?.('[data-tip]') as HTMLElement | null;
    if (el === current) return;
    hide();
    if (!el || !el.dataset.tip) return;
    current = el;
    timer = window.setTimeout(() => current === el && show(el), 350);
  });
  document.addEventListener('mousedown', hide, true);
  document.addEventListener('wheel', hide, { passive: true, capture: true });
  window.addEventListener('blur', hide);
}

/** Attach button tips inside a container and drop native titles so they don't double up. */
export function applyButtonTips(root: HTMLElement) {
  for (const b of Array.from(root.querySelectorAll('button'))) {
    if (b.closest('.seg')) {
      b.removeAttribute('title');
      continue;
    }
    const t = b.textContent?.trim() ?? '';
    const hit = BUTTON_TIPS.find(([k]) => t.startsWith(k));
    if (hit) {
      b.dataset.tip = hit[1];
      b.removeAttribute('title');
    }
  }
}
