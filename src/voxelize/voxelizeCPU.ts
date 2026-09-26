import type { GridDims } from '../solver/types';
import type { MeshData } from './mesh';
import type { VoxelInfo } from './VoxelizerGPU';

/**
 * CPU port of the GPU voxelizer (used by the WebGL2 fallback, which only needs the 2D slice).
 * Same algorithm: signed ray crossings along x, y, z → winding numbers → 2-of-3 vote, plus a thin
 * surface shell for parts too thin to own an interior cell.
 */
export function voxelizeCPU(mesh: MeshData | null, M: ArrayLike<number>, d: GridDims): { flags: Uint8Array; info: VoxelInfo } {
  const { nx, ny, nz } = d;
  const n = nx * ny * nz;
  const flags = new Uint8Array(n);
  if (!mesh || mesh.indices.length === 0) return { flags, info: { frontal: 0, solidCells: 0, min: [0, 0, 0], max: [0, 0, 0] } };
  const dims = [nx, ny, nz];
  const cell = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  const P = mesh.positions;
  const nv = P.length / 3;
  const T = new Float32Array(P.length);
  for (let i = 0; i < nv; i++) {
    const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
    T[3 * i] = M[0] * x + M[4] * y + M[8] * z + M[12];
    T[3 * i + 1] = M[1] * x + M[5] * y + M[9] * z + M[13];
    T[3 * i + 2] = M[2] * x + M[6] * y + M[10] * z + M[14];
  }
  const idx = mesh.indices;
  const hits = [new Int16Array(n), new Int16Array(n), new Int16Array(n)];
  const vote = new Uint8Array(n);
  const p0 = [0, 0, 0], p1 = [0, 0, 0], p2 = [0, 0, 0];
  for (let t = 0; t < idx.length; t += 3) {
    for (let k = 0; k < 3; k++) {
      p0[k] = T[3 * idx[t] + k];
      p1[k] = T[3 * idx[t + 1] + k];
      p2[k] = T[3 * idx[t + 2] + k];
    }
    // quick reject for the 2D slab
    if (nz === 1 && (Math.min(p0[2], p1[2], p2[2]) > 1 || Math.max(p0[2], p1[2], p2[2]) < 0)) continue;
    for (let a = 0; a < 3; a++) {
      const ua = (a + 1) % 3, va = (a + 2) % 3;
      const Ax = p0[ua], Ay = p0[va], Bx = p1[ua], By = p1[va], Cx = p2[ua], Cy = p2[va];
      const area = (Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax);
      if (Math.abs(area) < 1e-12) continue;
      const sgn = area > 0 ? -1 : 1;
      const u0 = Math.max(Math.ceil(Math.min(Ax, Bx, Cx) - 0.5), 0);
      const u1 = Math.min(Math.floor(Math.max(Ax, Bx, Cx) - 0.5), dims[ua] - 1);
      const v0 = Math.max(Math.ceil(Math.min(Ay, By, Cy) - 0.5), 0);
      const v1 = Math.min(Math.floor(Math.max(Ay, By, Cy) - 0.5), dims[va] - 1);
      const inv = 1 / area;
      for (let iu = u0; iu <= u1; iu++) {
        for (let iv = v0; iv <= v1; iv++) {
          const px = iu + 0.5 + 1.37e-4, py = iv + 0.5 + 2.71e-4;
          const w0 = ((Bx - px) * (Cy - py) - (By - py) * (Cx - px)) * inv;
          const w1 = ((Cx - px) * (Ay - py) - (Cy - py) * (Ax - px)) * inv;
          const w2 = 1 - w0 - w1;
          if (w0 < 0 || w1 < 0 || w2 < 0) continue;
          const h = w0 * p0[a] + w1 * p1[a] + w2 * p2[a];
          let k = Math.floor(h - 0.5) + 1;
          if (k >= dims[a]) continue;
          k = Math.max(k, 0);
          const c = a === 0 ? cell(k, iu, iv) : a === 1 ? cell(iv, k, iu) : cell(iu, iv, k);
          hits[a][c] += sgn;
        }
      }
    }
    // thin surface shell
    const e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    const e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    const cr = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const cl = Math.hypot(cr[0], cr[1], cr[2]);
    if (cl < 1e-12) continue;
    const nrm = cr.map((v) => v / cl);
    const m = Math.max(Math.hypot(...e1), Math.hypot(...e2), Math.hypot(e2[0] - e1[0], e2[1] - e1[1], e2[2] - e1[2]));
    const steps = Math.min(Math.max(Math.ceil(m / 0.4), 1), 384);
    let tol = 0.5 * Math.max(Math.abs(nrm[0]), Math.abs(nrm[1]), Math.abs(nrm[2]));
    if (nz === 1) {
      if (Math.hypot(nrm[0], nrm[1]) < 1e-4) continue;
      tol = 0.5 * Math.max(Math.abs(nrm[0]), Math.abs(nrm[1]));
    }
    for (let i = 0; i <= steps; i++) {
      for (let j = 0; j <= steps - i; j++) {
        const a = i / steps, b = j / steps;
        const x = p0[0] + e1[0] * a + e2[0] * b, y = p0[1] + e1[1] * a + e2[1] * b, z = p0[2] + e1[2] * a + e2[2] * b;
        if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) continue;
        const cx = Math.floor(x), cy = Math.floor(y), cz = Math.floor(z);
        const ctr = [cx + 0.5, cy + 0.5, nz === 1 ? z : cz + 0.5];
        const dd = (ctr[0] - p0[0]) * nrm[0] + (ctr[1] - p0[1]) * nrm[1] + (ctr[2] - p0[2]) * nrm[2];
        if (Math.abs(dd) > tol) continue;
        vote[cell(cx, cy, cz)] |= 8;
      }
    }
  }
  // scans
  for (let a = 0; a < 3; a++) {
    const ua = (a + 1) % 3, va = (a + 2) % 3;
    for (let iv = 0; iv < dims[va]; iv++) {
      for (let iu = 0; iu < dims[ua]; iu++) {
        let s = 0;
        for (let k = 0; k < dims[a]; k++) {
          const c = a === 0 ? cell(k, iu, iv) : a === 1 ? cell(iv, k, iu) : cell(iu, iv, k);
          s += hits[a][c];
          if (s !== 0) vote[c] |= 1 << a;
        }
      }
    }
  }
  const votes = (v: number) => ((v & 1) + ((v >> 1) & 1) + ((v >> 2) & 1)) >= 2;
  let solidCells = 0;
  const mn = [Infinity, Infinity, Infinity], mx = [-1, -1, -1];
  for (let z = 0; z < nz; z++)
    for (let y = 0; y < ny; y++)
      for (let x = 0; x < nx; x++) {
        const c = cell(x, y, z);
        const v = vote[c];
        let solid = votes(v);
        if (!solid && v & 8) {
          const near =
            (x > 0 && votes(vote[c - 1])) || (x + 1 < nx && votes(vote[c + 1])) ||
            (y > 0 && votes(vote[c - nx])) || (y + 1 < ny && votes(vote[c + nx])) ||
            (z > 0 && votes(vote[c - nx * ny])) || (z + 1 < nz && votes(vote[c + nx * ny]));
          solid = !near;
        }
        if (solid) {
          flags[c] = 1;
          solidCells++;
          mn[0] = Math.min(mn[0], x); mn[1] = Math.min(mn[1], y); mn[2] = Math.min(mn[2], z);
          mx[0] = Math.max(mx[0], x); mx[1] = Math.max(mx[1], y); mx[2] = Math.max(mx[2], z);
        }
      }
  let frontal = 0;
  for (let z = 0; z < nz; z++)
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) if (flags[cell(x, y, z)]) { frontal++; break; }
    }
  const ok = solidCells > 0;
  return {
    flags,
    info: { frontal, solidCells, min: ok ? (mn as [number, number, number]) : [0, 0, 0], max: ok ? (mx.map((v) => v + 1) as [number, number, number]) : [0, 0, 0] },
  };
}
