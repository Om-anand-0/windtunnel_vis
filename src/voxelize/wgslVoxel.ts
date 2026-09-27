/**
 * GPU voxelizer: 3-axis signed ray-crossing (winding number) + surface shell, majority vote.
 * All coordinates are in grid space (cell k spans [k, k+1], centre at k + 0.5).
 */
const COMMON = /* wgsl */ `
struct VP {
  M: mat4x4<f32>,
  nx: u32, ny: u32, nz: u32, n: u32,
  triCount: u32, axis: u32, strideX: u32, pad: u32,
};
@group(0) @binding(0) var<uniform> V: VP;

fn dimOf(a: u32) -> u32 {
  if (a == 0u) { return V.nx; }
  if (a == 1u) { return V.ny; }
  return V.nz;
}
fn cellOf(c: vec3<u32>) -> u32 {
  return c.x + V.nx * (c.y + V.ny * c.z);
}
`;

const TRI = /* wgsl */ `
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read> index: array<u32>;
fn vtx(i: u32) -> vec3<f32> {
  let k = index[i];
  let p = vec3<f32>(pos[3u * k], pos[3u * k + 1u], pos[3u * k + 2u]);
  return (V.M * vec4<f32>(p, 1.0)).xyz;
}
`;

export const crossingsWGSL = /* wgsl */ `
${COMMON}
${TRI}
@group(0) @binding(3) var<storage, read_write> hits: array<atomic<i32>>;

fn comp(p: vec3<f32>, a: u32) -> f32 {
  if (a == 0u) { return p.x; }
  if (a == 1u) { return p.y; }
  return p.z;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x + gid.y * V.strideX;
  if (t >= V.triCount) { return; }
  let p0 = vtx(3u * t);
  let p1 = vtx(3u * t + 1u);
  let p2 = vtx(3u * t + 2u);
  for (var a = 0u; a < 3u; a++) {
    let ua = (a + 1u) % 3u;
    let va = (a + 2u) % 3u;
    let A = vec2<f32>(comp(p0, ua), comp(p0, va));
    let B = vec2<f32>(comp(p1, ua), comp(p1, va));
    let C = vec2<f32>(comp(p2, ua), comp(p2, va));
    let area = (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
    if (abs(area) < 1e-12) { continue; }
    let sgn = select(1, -1, area > 0.0);   // entering a closed outward-oriented surface → +1
    let nu = i32(dimOf(ua));
    let nv = i32(dimOf(va));
    let na = i32(dimOf(a));
    let lo = min(min(A, B), C);
    let hi = max(max(A, B), C);
    let u0 = max(i32(ceil(lo.x - 0.5)), 0);
    let u1 = min(i32(floor(hi.x - 0.5)), nu - 1);
    let v0 = max(i32(ceil(lo.y - 0.5)), 0);
    let v1 = min(i32(floor(hi.y - 0.5)), nv - 1);
    let ha = comp(p0, a);
    let hb = comp(p1, a);
    let hc = comp(p2, a);
    let inv = 1.0 / area;
    for (var iu = u0; iu <= u1; iu++) {
      for (var iv = v0; iv <= v1; iv++) {
        // tiny irrational offset avoids double counting on shared edges / vertices
        let P = vec2<f32>(f32(iu) + 0.5 + 1.37e-4, f32(iv) + 0.5 + 2.71e-4);
        let w0 = ((B.x - P.x) * (C.y - P.y) - (B.y - P.y) * (C.x - P.x)) * inv;
        let w1 = ((C.x - P.x) * (A.y - P.y) - (C.y - P.y) * (A.x - P.x)) * inv;
        let w2 = 1.0 - w0 - w1;
        if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0) { continue; }
        let h = w0 * ha + w1 * hb + w2 * hc;
        var k = i32(floor(h - 0.5)) + 1;
        if (k >= na) { continue; }
        k = max(k, 0);
        var c = vec3<u32>(0u);
        if (a == 0u) { c = vec3<u32>(u32(k), u32(iu), u32(iv)); }
        else if (a == 1u) { c = vec3<u32>(u32(iv), u32(k), u32(iu)); }
        else { c = vec3<u32>(u32(iu), u32(iv), u32(k)); }
        atomicAdd(&hits[a * V.n + cellOf(c)], sgn);
      }
    }
  }
}
`;

export const scanWGSL = /* wgsl */ `
${COMMON}
@group(0) @binding(3) var<storage, read_write> hits: array<i32>;
@group(0) @binding(4) var<storage, read_write> vote: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let a = V.axis;
  let ua = (a + 1u) % 3u;
  let va = (a + 2u) % 3u;
  let nu = dimOf(ua);
  let nv = dimOf(va);
  let col = gid.x + gid.y * V.strideX;
  if (col >= nu * nv) { return; }
  let iu = col % nu;
  let iv = col / nu;
  var s = 0;
  let na = dimOf(a);
  for (var k = 0u; k < na; k++) {
    var c = vec3<u32>(0u);
    if (a == 0u) { c = vec3<u32>(k, iu, iv); }
    else if (a == 1u) { c = vec3<u32>(iv, k, iu); }
    else { c = vec3<u32>(iu, iv, k); }
    let id = cellOf(c);
    s += hits[a * V.n + id];
    if (s != 0) { atomicOr(&vote[id], 1u << a); }
  }
}
`;

export const surfaceWGSL = /* wgsl */ `
${COMMON}
${TRI}
@group(0) @binding(4) var<storage, read_write> vote: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x + gid.y * V.strideX;
  if (t >= V.triCount) { return; }
  let p0 = vtx(3u * t);
  let e1 = vtx(3u * t + 1u) - p0;
  let e2 = vtx(3u * t + 2u) - p0;
  let m = max(max(length(e1), length(e2)), length(e2 - e1));
  let steps = u32(clamp(ceil(m / 0.4), 1.0, 384.0));
  let dims = vec3<f32>(f32(V.nx), f32(V.ny), f32(V.nz));
  let fs = f32(steps);
  let cr = cross(e1, e2);
  let cl = length(cr);
  if (cl < 1e-12) { return; }
  let nrm = cr / cl;
  // thin (6-separating) surface voxelization: keep cells whose centre lies within half a cell of the
  // triangle plane measured in the max-norm. Prevents thin parts vanishing without inflating the body.
  let tol = 0.5 * max(max(abs(nrm.x), abs(nrm.y)), abs(nrm.z));
  let is2D = V.nz == 1u;
  for (var i = 0u; i <= steps; i++) {
    for (var j = 0u; j <= steps - i; j++) {
      var p = p0 + e1 * (f32(i) / fs) + e2 * (f32(j) / fs);
      if (any(p < vec3<f32>(0.0)) || any(p >= dims)) { continue; }
      let c = floor(p);
      var ctr = c + vec3<f32>(0.5);
      var t = tol;
      if (is2D) {
        // the 2D grid is a single plane: test distance of the cross-section line to the centre in-plane
        let n2 = length(nrm.xy);
        if (n2 < 1e-4) { continue; }
        ctr.z = p.z;
        t = 0.5 * max(abs(nrm.x), abs(nrm.y)) / n2 * n2;
        if (abs(p.z - 0.5) > 0.5) { continue; }
      }
      if (abs(dot(ctr - p0, nrm)) > t) { continue; }
      atomicOr(&vote[cellOf(vec3<u32>(c))], 8u);
    }
  }
}
`;

export const finalizeWGSL = /* wgsl */ `
${COMMON}
@group(0) @binding(4) var<storage, read_write> vote: array<u32>;
@group(0) @binding(5) var<storage, read_write> flags: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x + gid.y * V.strideX;
  if (id >= V.n) { return; }
  let v = vote[id];
  var solid = countOneBits(v & 7u) >= 2u;
  if (!solid && (v & 8u) != 0u) {
    // shell-only cell: keep it only where the part is too thin to own any interior cell nearby,
    // otherwise it would just inflate thick bodies by half a cell
    let x = id % V.nx;
    let y = (id / V.nx) % V.ny;
    let z = id / (V.nx * V.ny);
    var near = false;
    if (x > 0u) { near = near || countOneBits(vote[id - 1u] & 7u) >= 2u; }
    if (x + 1u < V.nx) { near = near || countOneBits(vote[id + 1u] & 7u) >= 2u; }
    if (y > 0u) { near = near || countOneBits(vote[id - V.nx] & 7u) >= 2u; }
    if (y + 1u < V.ny) { near = near || countOneBits(vote[id + V.nx] & 7u) >= 2u; }
    if (z > 0u) { near = near || countOneBits(vote[id - V.nx * V.ny] & 7u) >= 2u; }
    if (z + 1u < V.nz) { near = near || countOneBits(vote[id + V.nx * V.ny] & 7u) >= 2u; }
    solid = !near;
  }
  flags[id] = select(0u, 1u, solid);
}
`;

/**
 * Unsigned distance (cells) from cell centres near the surface to the mesh, for interpolated
 * bounce-back. Stored as ~bitcast<u32>(d) with atomicMax (0 = unknown), so a cleared buffer works.
 */
export const sdfWGSL = /* wgsl */ `
${COMMON}
${TRI}
@group(0) @binding(8) var<storage, read_write> sdf: array<atomic<u32>>;

fn closestPt(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
  let ab = b - a; let ac = c - a; let ap = p - a;
  let d1 = dot(ab, ap); let d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) { return a; }
  let bp = p - b; let d3 = dot(ab, bp); let d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) { return b; }
  let vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) { return a + ab * (d1 / (d1 - d3)); }
  let cp = p - c; let d5 = dot(ab, cp); let d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) { return c; }
  let vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) { return a + ac * (d2 / (d2 - d6)); }
  let va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) { return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))); }
  let den = 1.0 / (va + vb + vc);
  return a + ab * (vb * den) + ac * (vc * den);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x + gid.y * V.strideX;
  if (t >= V.triCount) { return; }
  let a = vtx(3u * t);
  let b = vtx(3u * t + 1u);
  let c = vtx(3u * t + 2u);
  let e1 = b - a;
  let e2 = c - a;
  if (length(cross(e1, e2)) < 1e-12) { return; }
  let m = max(max(length(e1), length(e2)), length(e2 - e1));
  let steps = u32(clamp(ceil(m / 0.7), 1.0, 256.0));
  let fs = f32(steps);
  let is2D = V.nz == 1u;
  let R = 2;
  for (var i = 0u; i <= steps; i++) {
    for (var j = 0u; j <= steps - i; j++) {
      let p = a + e1 * (f32(i) / fs) + e2 * (f32(j) / fs);
      if (is2D && abs(p.z - 0.5) > 2.5) { continue; }
      let base = vec3<i32>(floor(p));
      for (var dz = -R; dz <= R; dz++) {
        if (is2D && dz != 0) { continue; }
        for (var dy = -R; dy <= R; dy++) {
          for (var dx = -R; dx <= R; dx++) {
            var q = base + vec3<i32>(dx, dy, dz);
            if (is2D) { q.z = 0; }
            if (any(q < vec3<i32>(0)) || q.x >= i32(V.nx) || q.y >= i32(V.ny) || q.z >= i32(V.nz)) { continue; }
            var ctr = vec3<f32>(q) + vec3<f32>(0.5);
            let d = distance(ctr, closestPt(ctr, a, b, c));
            if (d > 2.5) { continue; }
            atomicMax(&sdf[cellOf(vec3<u32>(q))], ~bitcast<u32>(d));
          }
        }
      }
    }
  }
}
`;

/** Tag solid cells that belong to a wheel (flags = 1 | (k+1) << 8) for the rotating-wheel boundary. */
export const wheelTagWGSL = /* wgsl */ `
${COMMON}
struct Wheels { count: u32, p0: u32, p1: u32, p2: u32, w: array<vec4<f32>, 32> };
@group(0) @binding(5) var<storage, read_write> flags: array<u32>;
@group(0) @binding(7) var<uniform> WH: Wheels;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x + gid.y * V.strideX;
  if (id >= V.n || flags[id] == 0u) { return; }
  let p = vec3<f32>(f32(id % V.nx), f32((id / V.nx) % V.ny), f32(id / (V.nx * V.ny))) + vec3<f32>(0.5);
  for (var k = 0u; k < WH.count; k++) {
    let a = WH.w[2u * k];
    let b = WH.w[2u * k + 1u];
    let d = p - a.xyz;
    let al = dot(d, b.xyz);
    let rad = length(d - al * b.xyz);
    if (abs(al) <= b.w + 0.6 && rad <= 1.0 / a.w + 0.75) {
      flags[id] = 1u | ((k + 1u) << 8u);
      return;
    }
  }
}
`;

/**
 * Frontal projection + bounding box. info = [frontalCount, solidCount, minx, miny, minz, maxx, maxy, maxz]
 */
export const projectWGSL = /* wgsl */ `
${COMMON}
@group(0) @binding(5) var<storage, read> flags: array<u32>;
@group(0) @binding(6) var<storage, read_write> info: array<atomic<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x + gid.y * V.strideX;
  if (col >= V.ny * V.nz) { return; }
  let y = col % V.ny;
  let z = col / V.ny;
  var hit = false;
  var cnt = 0u;
  for (var x = 0u; x < V.nx; x++) {
    if (flags[cellOf(vec3<u32>(x, y, z))] != 0u) {
      if (!hit) { atomicMin(&info[2], x); }
      atomicMax(&info[5], x);
      hit = true;
      cnt++;
    }
  }
  if (hit) {
    atomicAdd(&info[0], 1u);
    atomicAdd(&info[1], cnt);
    atomicMin(&info[3], y); atomicMax(&info[6], y);
    atomicMin(&info[4], z); atomicMax(&info[7], z);
  }
}
`;
