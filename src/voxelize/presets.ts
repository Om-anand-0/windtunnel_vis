import * as THREE from 'three';
import { mergeObject, MeshData, normalizeMesh } from './mesh';

/**
 * Procedural, unbranded vehicle presets. Bodies are lofted superellipse cross-sections (closed,
 * outward oriented, watertight); wheels, wings and trailers are separate closed parts whose union
 * the voxelizer resolves with the winding-number rule.
 */

export interface PresetInfo {
  id: string;
  name: string;
  /** real-world reference length in metres */
  lengthM: number;
  placement: 'ground' | 'center';
  spanwise?: boolean;
  validation?: boolean;
}

export const PRESETS: PresetInfo[] = [
  { id: 'sedan', name: 'Sedan', lengthM: 4.8, placement: 'ground' },
  { id: 'sports', name: 'Sports car', lengthM: 4.5, placement: 'ground' },
  { id: 'suv', name: 'SUV / van', lengthM: 4.9, placement: 'ground' },
  { id: 'truck', name: 'Truck + trailer', lengthM: 16.5, placement: 'ground' },
  { id: 'f1', name: 'Open-wheel racer', lengthM: 5.5, placement: 'ground' },
  { id: 'sphere', name: 'Sphere (validation)', lengthM: 0.2, placement: 'center', validation: true },
  { id: 'cylinder', name: 'Cylinder (validation)', lengthM: 0.1, placement: 'center', spanwise: true, validation: true },
];

type Curve = [number, number][];

/** Monotone piecewise-cubic (Fritsch–Carlson) interpolation through control points. */
function interp(pts: Curve): (x: number) => number {
  const n = pts.length;
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const d: number[] = [];
  const m: number[] = new Array(n).fill(0);
  for (let i = 0; i < n - 1; i++) d.push((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]));
  m[0] = d[0];
  m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) m[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
  for (let i = 0; i < n - 1; i++) {
    if (d[i] === 0) { m[i] = 0; m[i + 1] = 0; continue; }
    const a = m[i] / d[i], b = m[i + 1] / d[i];
    const s = a * a + b * b;
    if (s > 9) { const t = 3 / Math.sqrt(s); m[i] = t * a * d[i]; m[i + 1] = t * b * d[i]; }
  }
  return (x: number) => {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let i = 0;
    while (x > xs[i + 1]) i++;
    const h = xs[i + 1] - xs[i];
    const t = (x - xs[i]) / h;
    const t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h * m[i] + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h * m[i + 1];
  };
}

interface LoftSpec {
  x0: number;
  x1: number;
  top: Curve;
  bot: Curve;
  belt: Curve;
  halfW: Curve;
  /** superellipse exponent (2 = ellipse, higher = boxier) */
  exp: number;
  expTop?: number;
  /** relative narrowing of the greenhouse at the roof */
  tumble: number;
  stations?: number;
  ring?: number;
  zOffset?: number;
}

/** Lofted closed body. Stations are clustered towards both ends to keep noses/tails smooth. */
function loft(spec: LoftSpec): THREE.BufferGeometry {
  const S = spec.stations ?? 120;
  const R = spec.ring ?? 64;
  const top = interp(spec.top), bot = interp(spec.bot), belt = interp(spec.belt), hw = interp(spec.halfW);
  const pos: number[] = [];
  const idx: number[] = [];
  const zo = spec.zOffset ?? 0;
  const se = (c: number, e: number) => Math.sign(c) * Math.pow(Math.abs(c), 2 / e);
  for (let s = 0; s <= S; s++) {
    const u = s / S;
    const t = 0.5 - 0.5 * Math.cos(Math.PI * u);
    const x = spec.x0 + (spec.x1 - spec.x0) * t;
    const yt = top(x), yb = bot(x);
    const yc = Math.min(Math.max(belt(x), yb + 1e-4), yt - 1e-4);
    const w = Math.max(hw(x), 1e-4);
    for (let r = 0; r < R; r++) {
      const phi = (2 * Math.PI * r) / R;
      const cyRaw = Math.sin(phi);
      const upper = cyRaw > 0;
      const e = upper ? spec.expTop ?? spec.exp : spec.exp;
      const cz = se(Math.cos(phi), e);
      const cy = se(cyRaw, e);
      const y = yc + cy * (upper ? yt - yc : yc - yb);
      const z = w * cz * (upper ? 1 - spec.tumble * cy : 1);
      pos.push(x, y, z + zo);
    }
  }
  const ring = (s: number, r: number) => s * R + (r % R);
  for (let s = 0; s < S; s++) {
    for (let r = 0; r < R; r++) {
      const a = ring(s, r), b = ring(s, r + 1), c = ring(s + 1, r), d = ring(s + 1, r + 1);
      idx.push(a, c, b, b, c, d);
    }
  }
  // end caps
  const capCentre = (s: number) => {
    let cx = 0, cy = 0, cz = 0;
    for (let r = 0; r < R; r++) { const k = ring(s, r) * 3; cx += pos[k]; cy += pos[k + 1]; cz += pos[k + 2]; }
    pos.push(cx / R, cy / R, cz / R);
    return pos.length / 3 - 1;
  };
  const c0 = capCentre(0);
  const c1 = capCentre(S);
  for (let r = 0; r < R; r++) {
    idx.push(c0, ring(0, r), ring(0, r + 1));
    idx.push(c1, ring(S, r + 1), ring(S, r));
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

function wheel(r: number, width: number, x: number, z: number, y = r): THREE.Mesh {
  const g = new THREE.CylinderGeometry(r, r, width, 40, 1, false);
  g.rotateX(Math.PI / 2);
  g.translate(x, y, z);
  return new THREE.Mesh(g);
}

function roundedBox(x0: number, x1: number, y0: number, y1: number, halfW: number, exp = 8, zOffset = 0): THREE.BufferGeometry {
  const L = x1 - x0;
  const r = Math.min(0.04 * L, (y1 - y0) * 0.1);
  return loft({
    x0, x1,
    top: [[x0, y0 + (y1 - y0) * 0.6], [x0 + r * 0.5, y1 - r * 0.3], [x0 + r * 2, y1], [x1 - r, y1], [x1, y1 - r * 0.5]],
    bot: [[x0, y0 + (y1 - y0) * 0.3], [x0 + r * 0.5, y0 + r * 0.2], [x0 + r * 2, y0], [x1, y0]],
    belt: [[x0, (y0 + y1) / 2], [x1, (y0 + y1) / 2]],
    halfW: [[x0, halfW * 0.8], [x0 + r * 1.5, halfW], [x1 - r, halfW], [x1, halfW * 0.92]],
    exp, tumble: 0, stations: 60, ring: 48, zOffset,
  });
}

/** NACA 4-digit airfoil shape (chord along x), optionally rotated by an angle of attack. */
function airfoil(chord: number, camber: number, camberPos: number, thick: number, aoaDeg: number): THREE.Shape {
  const N = 40;
  const up: [number, number][] = [];
  const lo: [number, number][] = [];
  for (let i = 0; i <= N; i++) {
    const b = (i / N) * Math.PI;
    const x = 0.5 - 0.5 * Math.cos(b);
    const yt = 5 * thick * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4);
    let yc = 0, dyc = 0;
    if (camber > 0) {
      if (x < camberPos) { yc = (camber / camberPos ** 2) * (2 * camberPos * x - x * x); dyc = ((2 * camber) / camberPos ** 2) * (camberPos - x); }
      else { yc = (camber / (1 - camberPos) ** 2) * (1 - 2 * camberPos + 2 * camberPos * x - x * x); dyc = ((2 * camber) / (1 - camberPos) ** 2) * (camberPos - x); }
    }
    const th = Math.atan(dyc);
    up.push([x - yt * Math.sin(th), yc + yt * Math.cos(th)]);
    lo.push([x + yt * Math.sin(th), yc - yt * Math.cos(th)]);
  }
  const a = (aoaDeg * Math.PI) / 180;
  const tr = ([x, y]: [number, number]): [number, number] => {
    const px = (x - 0.25) * chord, py = y * chord;
    return [px * Math.cos(a) + py * Math.sin(a), -px * Math.sin(a) + py * Math.cos(a)];
  };
  const shape = new THREE.Shape();
  const pts = [...up.map(tr), ...lo.slice(1, -1).reverse().map(tr)];
  shape.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) shape.lineTo(pts[i][0], pts[i][1]);
  shape.closePath();
  return shape;
}

/** Wing element: inverted (downforce) airfoil extruded along z, centred at (x,y). */
function wing(x: number, y: number, chord: number, span: number, aoaDeg: number, camber = 0.06): THREE.Mesh {
  const s = airfoil(chord, camber, 0.4, 0.12, aoaDeg);
  const g = new THREE.ExtrudeGeometry(s, { depth: span, bevelEnabled: false, curveSegments: 1 });
  g.scale(1, -1, 1); // invert → downforce
  g.translate(x, y, -span / 2);
  return new THREE.Mesh(g);
}

function plate(x0: number, x1: number, y0: number, y1: number, z0: number, z1: number): THREE.Mesh {
  const g = new THREE.BoxGeometry(x1 - x0, y1 - y0, z1 - z0);
  g.translate((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
  return new THREE.Mesh(g);
}

function sedan(): THREE.Object3D {
  const o = new THREE.Group();
  o.add(new THREE.Mesh(loft({
    x0: 0, x1: 1,
    top: [[0, 0.085], [0.015, 0.125], [0.06, 0.16], [0.27, 0.183], [0.31, 0.19], [0.44, 0.292], [0.5, 0.302], [0.63, 0.298], [0.76, 0.255], [0.84, 0.21], [0.96, 0.205], [1, 0.165]],
    bot: [[0, 0.06], [0.03, 0.04], [0.1, 0.032], [0.9, 0.036], [0.97, 0.06], [1, 0.1]],
    belt: [[0, 0.07], [0.1, 0.13], [0.5, 0.15], [0.9, 0.16], [1, 0.14]],
    halfW: [[0, 0.1], [0.04, 0.165], [0.15, 0.19], [0.85, 0.19], [0.97, 0.175], [1, 0.14]],
    exp: 3.2, expTop: 2.6, tumble: 0.32,
  })));
  for (const x of [0.165, 0.79]) for (const z of [-0.158, 0.158]) o.add(wheel(0.068, 0.048, x, z));
  return o;
}

function sports(): THREE.Object3D {
  const o = new THREE.Group();
  o.add(new THREE.Mesh(loft({
    x0: 0, x1: 1,
    top: [[0, 0.06], [0.015, 0.09], [0.08, 0.12], [0.3, 0.15], [0.35, 0.162], [0.47, 0.24], [0.55, 0.252], [0.66, 0.235], [0.85, 0.19], [0.96, 0.17], [1, 0.14]],
    bot: [[0, 0.045], [0.03, 0.028], [0.1, 0.024], [0.9, 0.026], [0.98, 0.06], [1, 0.08]],
    belt: [[0, 0.05], [0.12, 0.1], [0.5, 0.12], [0.85, 0.135], [1, 0.12]],
    halfW: [[0, 0.1], [0.05, 0.185], [0.2, 0.205], [0.75, 0.215], [0.95, 0.2], [1, 0.16]],
    exp: 3.0, expTop: 2.3, tumble: 0.4,
  })));
  for (const [x, r] of [[0.17, 0.07], [0.79, 0.074]] as const) for (const z of [-0.172, 0.172]) o.add(wheel(r, 0.058, x, z));
  return o;
}

function suv(): THREE.Object3D {
  const o = new THREE.Group();
  o.add(new THREE.Mesh(loft({
    x0: 0, x1: 1,
    top: [[0, 0.13], [0.012, 0.175], [0.05, 0.21], [0.24, 0.235], [0.29, 0.25], [0.4, 0.352], [0.46, 0.366], [0.93, 0.362], [0.975, 0.34], [1, 0.25]],
    bot: [[0, 0.09], [0.03, 0.055], [0.1, 0.048], [0.9, 0.05], [0.98, 0.07], [1, 0.1]],
    belt: [[0, 0.12], [0.1, 0.19], [0.9, 0.21], [1, 0.2]],
    halfW: [[0, 0.13], [0.03, 0.175], [0.12, 0.192], [0.9, 0.192], [0.99, 0.18], [1, 0.16]],
    exp: 4.5, expTop: 3.2, tumble: 0.18,
  })));
  for (const x of [0.17, 0.8]) for (const z of [-0.158, 0.158]) o.add(wheel(0.078, 0.052, x, z));
  return o;
}

function truck(): THREE.Object3D {
  const o = new THREE.Group();
  // cab with roof fairing
  o.add(new THREE.Mesh(loft({
    x0: 0, x1: 0.15,
    top: [[0, 0.1], [0.01, 0.17], [0.03, 0.195], [0.07, 0.205], [0.1, 0.228], [0.13, 0.238], [0.15, 0.236]],
    bot: [[0, 0.045], [0.02, 0.03], [0.15, 0.03]],
    belt: [[0, 0.1], [0.15, 0.12]],
    halfW: [[0, 0.066], [0.015, 0.074], [0.15, 0.076]],
    exp: 7, expTop: 5, tumble: 0.05, stations: 60, ring: 56,
  })));
  // chassis rail under the gap
  o.add(plate(0.1, 0.2, 0.03, 0.055, -0.035, 0.035));
  // trailer
  o.add(new THREE.Mesh(roundedBox(0.17, 1.0, 0.075, 0.245, 0.076, 10)));
  for (const x of [0.07]) for (const z of [-0.058, 0.058]) o.add(wheel(0.031, 0.02, x, z));
  for (const x of [0.17, 0.23, 0.87, 0.92, 0.97]) for (const z of [-0.058, 0.058]) o.add(wheel(0.031, 0.024, x, z));
  return o;
}

function f1(): THREE.Object3D {
  const o = new THREE.Group();
  // main chassis / nose / engine cover
  o.add(new THREE.Mesh(loft({
    x0: 0.03, x1: 0.93,
    top: [[0.03, 0.045], [0.06, 0.06], [0.15, 0.078], [0.3, 0.1], [0.36, 0.108], [0.4, 0.172], [0.46, 0.17], [0.62, 0.12], [0.8, 0.085], [0.93, 0.065]],
    bot: [[0.03, 0.04], [0.06, 0.035], [0.15, 0.04], [0.28, 0.018], [0.85, 0.018], [0.93, 0.03]],
    belt: [[0.03, 0.043], [0.15, 0.06], [0.35, 0.075], [0.6, 0.06], [0.93, 0.045]],
    halfW: [[0.03, 0.01], [0.08, 0.02], [0.2, 0.035], [0.3, 0.05], [0.4, 0.052], [0.6, 0.04], [0.85, 0.028], [0.93, 0.02]],
    exp: 2.6, expTop: 2.2, tumble: 0.3,
  })));
  // sidepods
  for (const z of [-0.085, 0.085]) {
    o.add(new THREE.Mesh(loft({
      x0: 0.33, x1: 0.72,
      top: [[0.33, 0.06], [0.36, 0.085], [0.45, 0.09], [0.6, 0.07], [0.72, 0.035]],
      bot: [[0.33, 0.03], [0.36, 0.022], [0.72, 0.022]],
      belt: [[0.33, 0.05], [0.72, 0.03]],
      halfW: [[0.33, 0.02], [0.36, 0.042], [0.5, 0.04], [0.72, 0.012]],
      exp: 3, tumble: 0.2, stations: 50, ring: 40, zOffset: z,
    })));
  }
  // floor
  o.add(plate(0.3, 0.86, 0.012, 0.02, -0.135, 0.135));
  // front wing (two elements) + endplates
  o.add(wing(0.03, 0.028, 0.07, 0.36, -4));
  o.add(wing(0.085, 0.045, 0.04, 0.34, -18));
  for (const z of [-0.18, 0.18]) o.add(plate(0.0, 0.11, 0.012, 0.065, z - 0.003, z + 0.003));
  // rear wing (main plane + flap) + endplates + pylon
  o.add(wing(0.9, 0.16, 0.06, 0.18, -8));
  o.add(wing(0.945, 0.18, 0.035, 0.18, -25));
  for (const z of [-0.09, 0.09]) o.add(plate(0.86, 0.98, 0.09, 0.2, z - 0.003, z + 0.003));
  o.add(plate(0.9, 0.93, 0.06, 0.16, -0.004, 0.004));
  // wheels
  for (const z of [-0.145, 0.145]) {
    o.add(wheel(0.06, 0.055, 0.17, z));
    o.add(wheel(0.062, 0.072, 0.77, z));
  }
  return o;
}

function sphere(): THREE.Object3D {
  return new THREE.Mesh(new THREE.SphereGeometry(0.5, 96, 64));
}

function cylinder(): THREE.Object3D {
  const g = new THREE.CylinderGeometry(0.5, 0.5, 1, 128, 1, false);
  g.rotateX(Math.PI / 2);
  return new THREE.Mesh(g);
}

const BUILDERS: Record<string, () => THREE.Object3D> = { sedan, sports, suv, truck, f1, sphere, cylinder };

const cache = new Map<string, MeshData>();

export function buildPreset(id: string): MeshData {
  const hit = cache.get(id);
  if (hit) return hit;
  const obj = BUILDERS[id]();
  const raw = mergeObject(obj);
  const mesh = normalizeMesh(raw, { alignLongest: id !== 'cylinder' });
  cache.set(id, mesh);
  return mesh;
}
