import type { GroundMode } from './solver/types';
import type { CameraPreset } from './render/camera';
import type { ColormapName } from './render/colormaps';

export type Quality = 'low' | 'medium' | 'high' | 'ultra' | 'max';
export const QUALITIES: Quality[] = ['low', 'medium', 'high', 'ultra', 'max'];

export const GRID_2D: Record<Quality, { nx: number; ny: number; nz: number }> = {
  low: { nx: 512, ny: 192, nz: 1 },
  medium: { nx: 768, ny: 288, nz: 1 },
  high: { nx: 1024, ny: 384, nz: 1 },
  ultra: { nx: 1536, ny: 576, nz: 1 },
  max: { nx: 2048, ny: 768, nz: 1 },
};

export const GRID_3D: Record<Quality, { nx: number; ny: number; nz: number }> = {
  low: { nx: 160, ny: 80, nz: 80 },
  medium: { nx: 224, ny: 96, nz: 112 },
  high: { nx: 256, ny: 128, nz: 128 },
  ultra: { nx: 320, ny: 160, nz: 160 },
  max: { nx: 384, ny: 176, nz: 192 },
};

export interface FieldDef {
  id: number;
  key: string;
  label: string;
  unit: string;
  cmap: ColormapName;
  range: [number, number];
  /** range in 3D if different */
  range3D?: [number, number];
}

export const FIELDS: FieldDef[] = [
  { id: 0, key: 'speed', label: 'Velocity magnitude', unit: '|u| / U∞', cmap: 'turbo', range: [0, 1.5] },
  { id: 1, key: 'ux', label: 'Streamwise velocity', unit: 'uₓ / U∞', cmap: 'coolwarm', range: [-0.6, 1.4] },
  { id: 2, key: 'cp', label: 'Pressure coefficient', unit: 'Cp', cmap: 'coolwarm', range: [-1.5, 1] },
  { id: 3, key: 'vort', label: 'Vorticity', unit: 'ω·L / U∞', cmap: 'coolwarm', range: [-60, 60], range3D: [-15, 15] },
  { id: 4, key: 'ti', label: 'Turbulence intensity', unit: "√(⅔k) / U∞", cmap: 'inferno', range: [0, 0.4] },
  { id: 5, key: 'q', label: 'Q-criterion', unit: 'Q·L² / U∞²', cmap: 'coolwarm', range: [-2000, 2000], range3D: [-300, 300] },
];

export interface Settings {
  mode: '2d' | '3d';
  quality2D: Quality;
  quality3D: Quality;
  /** population storage precision (3D default fp16: half the memory and bandwidth) */
  precision: 'f32' | 'f16';
  vehicle: string;
  upAxis: 'y' | 'z';
  flip: boolean;
  lengthM: number;
  speedKmh: number;
  /** kinematic viscosity in 1e-5 m²/s */
  nuAir: number;
  reOverrideOn: boolean;
  reOverride: number;
  yaw: number;
  pitch: number;
  rideMm: number;
  ground: GroundMode;
  rotatingWheels: boolean;
  /** interpolated (Bouzidi) bounce-back from the true surface distance */
  interpWalls: boolean;
  lesCs: number;
  paused: boolean;
  simSpeed: number;
  targetFps: number;

  sliceOn: boolean;
  sliceAxis: 0 | 1 | 2;
  field: number;
  cmap: ColormapName;
  slicePos: number;
  sliceMean: boolean;
  sliceOpacity: number;
  autoRange: boolean;
  vmin: number;
  vmax: number;
  recirc: boolean;
  qContour: boolean;

  bodyOn: boolean;
  surface: 'lit' | 'cp';
  surfaceMean: boolean;

  particlesOn: boolean;
  particleCount: number;
  trail: number;
  emitter: 'inlet' | 'rake';
  smokeColor: 'smoke' | 'speed';

  streamOn: boolean;
  seeds: number;
  streamMean: boolean;
  rakeOrient: 'vertical' | 'horizontal';
  rakeX: number;
  rakeY: number;
  rakeZ: number;
  rakeSpan: number;

  isoOn: boolean;
  isoMode: 'q' | 'recirc';
  isoThr: number;
  isoColor: 'speed' | 'rotation';

  camera: CameraPreset;
}

export function defaultSettings(): Settings {
  return {
    mode: '3d',
    quality2D: 'high',
    quality3D: 'high',
    precision: 'f16',
    vehicle: 'sedan',
    upAxis: 'y',
    flip: false,
    lengthM: 4.8,
    speedKmh: 120,
    nuAir: 1.5,
    reOverrideOn: false,
    reOverride: 20000,
    yaw: 0,
    pitch: 0,
    rideMm: 0,
    ground: 'moving',
    rotatingWheels: true,
    interpWalls: true,
    lesCs: 0.14,
    paused: false,
    simSpeed: 1,
    targetFps: 45,
    sliceOn: true,
    sliceAxis: 2,
    field: 0,
    cmap: 'turbo',
    slicePos: 0.5,
    sliceMean: false,
    sliceOpacity: 1,
    autoRange: true,
    vmin: 0,
    vmax: 1.5,
    recirc: false,
    qContour: false,
    bodyOn: true,
    surface: 'cp',
    surfaceMean: true,
    particlesOn: true,
    particleCount: 150000,
    trail: 20,
    emitter: 'rake',
    smokeColor: 'smoke',
    streamOn: true,
    seeds: 40,
    streamMean: false,
    rakeOrient: 'vertical',
    rakeX: 0.1,
    rakeY: 0.22,
    rakeZ: 0.5,
    rakeSpan: 0.4,
    isoOn: true,
    isoMode: 'q',
    isoThr: 200,
    isoColor: 'speed',
    camera: 'free',
  };
}
