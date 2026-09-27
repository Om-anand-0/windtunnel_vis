/** Everything the renderers need to draw one frame. Built by the app from the UI state. */
export interface RenderState {
  viewProj: Float32Array;
  invViewProj: Float32Array;
  eye: [number, number, number];
  time: number;
  groundOffset: number;
  flow: { U: number; rhoRef: number; Lref: number };
  box: boolean;
  ground: { moving: boolean };
  mesh: {
    visible: boolean;
    model: Float32Array;
    normalMatrix: Float32Array;
    mode: 'lit' | 'cp';
    cmap: number;
    useMean: boolean;
    cpMin: number;
    cpMax: number;
    opacity: number;
  };
  slice: {
    visible: boolean;
    /** 0: YZ (cross-flow), 1: XZ (horizontal), 2: XY (side) */
    axis: 0 | 1 | 2;
    /** 0 speed, 1 ux, 2 Cp, 3 vorticity, 4 TI, 5 Q */
    field: number;
    cmap: number;
    pos: number;
    vmin: number;
    vmax: number;
    opacity: number;
    mean: boolean;
    recirc: boolean;
    qContour: boolean;
    qThr: number;
  };
  volume: {
    visible: boolean;
    mode: 'q' | 'recirc';
    colorBy: 'speed' | 'rotation';
    thr: number;
    step: number;
    opacity: number;
  };
  particles: {
    enabled: boolean;
    count: number;
    trail: number;
    emitter: 'inlet' | 'rake';
    colorMode: 'smoke' | 'speed';
    width: number;
    alpha: number;
    advance: boolean;
    steps: number;
    maxAge: number;
    nozzles: number;
  };
  streamlines: {
    enabled: boolean;
    seeds: number;
    points: number;
    step: number;
    useMean: boolean;
    width: number;
    alpha: number;
  };
  rake: { a: [number, number, number]; b: [number, number, number]; visible: boolean; active: boolean };
  /** probe positions (grid coords) and marker colours */
  probes: { pos: [number, number, number]; color: [number, number, number] }[];
}
