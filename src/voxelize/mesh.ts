import * as THREE from 'three';

/** Indexed triangle mesh in normalized vehicle space (length 1 along +x, ground at y = 0, centred in x and z). */
export interface MeshData {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  /** bounding box after normalization */
  min: THREE.Vector3;
  max: THREE.Vector3;
  /** raw → normalized transform (positions) */
  transform?: THREE.Matrix4;
  /** wheels (normalized space) of procedural presets: they spin with the rolling road */
  wheels?: WheelDef[];
}

/** A wheel: cylinder with centre c, unit axis, radius r and half-width hw. */
export interface WheelDef {
  c: [number, number, number];
  axis: [number, number, number];
  r: number;
  hw: number;
}

/** Signed volume of an indexed triangle soup (positive when outward oriented). */
function signedVolume(pos: ArrayLike<number>, idx: ArrayLike<number>, start = 0, end = idx.length): number {
  let v = 0;
  for (let t = start; t < end; t += 3) {
    const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
    v += pos[a] * (pos[b + 1] * pos[c + 2] - pos[b + 2] * pos[c + 1])
      - pos[a + 1] * (pos[b] * pos[c + 2] - pos[b + 2] * pos[c])
      + pos[a + 2] * (pos[b] * pos[c + 1] - pos[b + 1] * pos[c]);
  }
  return v / 6;
}

/**
 * Merge every mesh under `root` (with world transforms applied) into one indexed geometry.
 * Each part is re-oriented to have positive signed volume so the winding-number voxelizer
 * treats overlapping parts as a union.
 */
export function mergeObject(root: THREE.Object3D): { positions: Float32Array; normals: Float32Array; indices: Uint32Array } {
  root.updateMatrixWorld(true);
  const parts: { p: Float32Array; n: Float32Array; i: Uint32Array }[] = [];
  root.traverse((o) => {
    const m = o as THREE.Mesh;
    if (!m.isMesh || !m.geometry) return;
    let g = m.geometry.clone();
    if (!g.getAttribute('position')) return;
    g.applyMatrix4(m.matrixWorld);
    // strip everything except position/normal to allow simple merging
    for (const name of Object.keys(g.attributes)) if (name !== 'position' && name !== 'normal') g.deleteAttribute(name);
    if (!g.index) {
      const count = g.getAttribute('position').count;
      const idx = new Uint32Array(count);
      for (let k = 0; k < count; k++) idx[k] = k;
      g.setIndex(new THREE.BufferAttribute(idx, 1));
    }
    if (!g.getAttribute('normal')) g.computeVertexNormals();
    const pa = g.getAttribute('position') as THREE.BufferAttribute;
    const na = g.getAttribute('normal') as THREE.BufferAttribute;
    const p = new Float32Array(pa.count * 3);
    const n = new Float32Array(pa.count * 3);
    for (let k = 0; k < pa.count; k++) {
      p[3 * k] = pa.getX(k); p[3 * k + 1] = pa.getY(k); p[3 * k + 2] = pa.getZ(k);
      n[3 * k] = na.getX(k); n[3 * k + 1] = na.getY(k); n[3 * k + 2] = na.getZ(k);
    }
    const src = g.index!.array;
    const i = new Uint32Array(src.length - (src.length % 3));
    for (let k = 0; k < i.length; k++) i[k] = src[k];
    if (signedVolume(p, i) < 0) {
      for (let k = 0; k < i.length; k += 3) {
        const t = i[k + 1]; i[k + 1] = i[k + 2]; i[k + 2] = t;
      }
      for (let k = 0; k < n.length; k++) n[k] = -n[k];
    }
    parts.push({ p, n, i });
    g.dispose();
    g = null as unknown as THREE.BufferGeometry;
  });
  let nv = 0, ni = 0;
  for (const part of parts) { nv += part.p.length; ni += part.i.length; }
  const positions = new Float32Array(nv);
  const normals = new Float32Array(nv);
  const indices = new Uint32Array(ni);
  let ov = 0, oi = 0;
  for (const part of parts) {
    positions.set(part.p, ov);
    normals.set(part.n, ov);
    const base = ov / 3;
    for (let k = 0; k < part.i.length; k++) indices[oi + k] = part.i[k] + base;
    ov += part.p.length;
    oi += part.i.length;
  }
  return { positions, normals, indices };
}

export type UpAxis = 'y' | 'z';

/**
 * Normalize a raw merged mesh: pick the up axis, align the longest horizontal extent with +x,
 * optionally flip the driving direction, scale the length to 1, centre in x/z and put the lowest
 * point on y = 0.
 */
export function normalizeMesh(
  raw: { positions: Float32Array; normals: Float32Array; indices: Uint32Array },
  opts: { up?: UpAxis; flip?: boolean; alignLongest?: boolean } = {},
): MeshData {
  const p = raw.positions.slice();
  const n = raw.normals.slice();
  const rot = new THREE.Matrix4();
  if (opts.up === 'z') rot.makeRotationX(-Math.PI / 2);
  const v = new THREE.Vector3();
  const total = new THREE.Matrix4();
  const apply = (m: THREE.Matrix4, normal: boolean) => {
    total.premultiply(m);
    const nm = new THREE.Matrix3().getNormalMatrix(m);
    for (let k = 0; k < p.length; k += 3) {
      v.set(p[k], p[k + 1], p[k + 2]).applyMatrix4(m);
      p[k] = v.x; p[k + 1] = v.y; p[k + 2] = v.z;
      if (normal) {
        v.set(n[k], n[k + 1], n[k + 2]).applyMatrix3(nm).normalize();
        n[k] = v.x; n[k + 1] = v.y; n[k + 2] = v.z;
      }
    }
  };
  apply(rot, true);
  const box = new THREE.Box3();
  const bb = () => {
    box.makeEmpty();
    for (let k = 0; k < p.length; k += 3) box.expandByPoint(v.set(p[k], p[k + 1], p[k + 2]));
    return box;
  };
  bb();
  if (opts.alignLongest !== false) {
    const sx = box.max.x - box.min.x, sz = box.max.z - box.min.z;
    if (sz > sx * 1.05) {
      apply(new THREE.Matrix4().makeRotationY(Math.PI / 2), true);
      bb();
    }
  }
  if (opts.flip) {
    apply(new THREE.Matrix4().makeRotationY(Math.PI), true);
    bb();
  }
  const len = Math.max(box.max.x - box.min.x, 1e-9);
  const cx = (box.max.x + box.min.x) / 2, cz = (box.max.z + box.min.z) / 2;
  const m = new THREE.Matrix4().makeScale(1 / len, 1 / len, 1 / len).multiply(new THREE.Matrix4().makeTranslation(-cx, -box.min.y, -cz));
  apply(m, false);
  bb();
  return { positions: p, normals: n, indices: raw.indices, min: box.min.clone(), max: box.max.clone(), transform: total };
}

/** Wheels in grid space (centre, axis, radius, half-width), for the rotating-wheel boundary. */
export function gridWheels(mesh: MeshData, M: THREE.Matrix4): WheelDef[] {
  if (!mesh.wheels) return [];
  const s = new THREE.Vector3();
  M.decompose(new THREE.Vector3(), new THREE.Quaternion(), s);
  const lin = new THREE.Matrix3().setFromMatrix4(M);
  return mesh.wheels.map((w) => {
    const c = new THREE.Vector3(...w.c).applyMatrix4(M);
    const a = new THREE.Vector3(...w.axis).applyMatrix3(lin);
    const axisScale = a.length();
    a.normalize();
    return { c: [c.x, c.y, c.z], axis: [a.x, a.y, a.z], r: w.r * s.x, hw: w.hw * axisScale };
  });
}

export interface Placement {
  /** 'ground': wheels on the floor, length = lengthFrac·nx. 'center': floating, diameter = diamFrac·ny */
  mode: 'ground' | 'center';
  lengthFrac: number;
  diamFrac: number;
  /** x position of the vehicle centre as a fraction of nx */
  xFrac: number;
  yawDeg: number;
  pitchDeg: number;
  /** ride height offset in cells */
  rideCells: number;
  /** stretch along z to span the whole tunnel (quasi-2D cylinder) */
  spanwise?: boolean;
}

/**
 * Build the mesh→grid matrix. Returns the matrix and the scale (cells per vehicle length).
 * In 2D (nz == 1) the grid plane z ∈ [0,1] is placed on the vehicle centre line.
 */
export function placementMatrix(mesh: MeshData, dims: { nx: number; ny: number; nz: number }, pl: Placement): { matrix: THREE.Matrix4; scale: number } {
  const { nx, ny, nz } = dims;
  const height = mesh.max.y - mesh.min.y;
  const scale = pl.mode === 'ground' ? pl.lengthFrac * nx : (pl.diamFrac * ny) / Math.max(height, 1e-6);
  const cy = (mesh.max.y + mesh.min.y) / 2;
  // rotate about the vehicle centre (at mid height)
  const R = new THREE.Matrix4()
    .makeRotationY((pl.yawDeg * Math.PI) / 180)
    .multiply(new THREE.Matrix4().makeRotationZ((-pl.pitchDeg * Math.PI) / 180));
  const S = new THREE.Matrix4().makeScale(scale, scale, pl.spanwise ? (nz > 1 ? (nz + 4) / Math.max(mesh.max.z - mesh.min.z, 1e-6) : scale) : scale);
  const C = new THREE.Matrix4().makeTranslation(0, -cy, 0);
  const M = R.clone().multiply(S).multiply(C);
  // find the lowest point after rotation to put the vehicle on the ground
  let minY = Infinity;
  const v = new THREE.Vector3();
  const p = mesh.positions;
  const stride = Math.max(3, Math.floor(p.length / 3 / 200000) * 3);
  for (let k = 0; k < p.length; k += stride) {
    v.set(p[k], p[k + 1], p[k + 2]).applyMatrix4(M);
    if (v.y < minY) minY = v.y;
  }
  const zc = nz > 1 ? nz / 2 : 0.5;
  const ty = pl.mode === 'ground' ? -minY + pl.rideCells : ny / 2 + 0.37;
  const T = new THREE.Matrix4().makeTranslation(pl.xFrac * nx, ty, zc);
  return { matrix: T.multiply(M), scale };
}

/** Projected frontal area of a mesh (normalized units², after yaw), rasterized on the CPU. */
export function frontalAreaNormalized(mesh: MeshData, yawDeg: number, pitchDeg: number, res = 256): number {
  const R = new THREE.Matrix4()
    .makeRotationY((yawDeg * Math.PI) / 180)
    .multiply(new THREE.Matrix4().makeRotationZ((-pitchDeg * Math.PI) / 180));
  const p = mesh.positions;
  const q = new Float32Array(p.length);
  const v = new THREE.Vector3();
  let y0 = Infinity, y1 = -Infinity, z0 = Infinity, z1 = -Infinity;
  for (let k = 0; k < p.length; k += 3) {
    v.set(p[k], p[k + 1], p[k + 2]).applyMatrix4(R);
    q[k] = v.x; q[k + 1] = v.y; q[k + 2] = v.z;
    y0 = Math.min(y0, v.y); y1 = Math.max(y1, v.y); z0 = Math.min(z0, v.z); z1 = Math.max(z1, v.z);
  }
  const span = Math.max(y1 - y0, z1 - z0) * 1.02;
  const h = span / res;
  const img = new Uint8Array(res * res);
  const idx = mesh.indices;
  for (let t = 0; t < idx.length; t += 3) {
    const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
    const ax = (q[a + 2] - z0) / h, ay = (q[a + 1] - y0) / h;
    const bx = (q[b + 2] - z0) / h, by = (q[b + 1] - y0) / h;
    const cx = (q[c + 2] - z0) / h, cy = (q[c + 1] - y0) / h;
    const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (Math.abs(area) < 1e-12) continue;
    const i0 = Math.max(0, Math.floor(Math.min(ax, bx, cx))), i1 = Math.min(res - 1, Math.ceil(Math.max(ax, bx, cx)));
    const j0 = Math.max(0, Math.floor(Math.min(ay, by, cy))), j1 = Math.min(res - 1, Math.ceil(Math.max(ay, by, cy)));
    for (let j = j0; j <= j1; j++) {
      for (let i = i0; i <= i1; i++) {
        const px = i + 0.5, py = j + 0.5;
        const w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area;
        const w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area;
        const w2 = 1 - w0 - w1;
        if (w0 >= 0 && w1 >= 0 && w2 >= 0) img[i + j * res] = 1;
      }
    }
  }
  let cnt = 0;
  for (let k = 0; k < img.length; k++) cnt += img[k];
  return cnt * h * h;
}
