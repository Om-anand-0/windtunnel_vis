import { fl, Lattice, tensorPairs } from './lattice';

/**
 * Shared uniform block for all solver kernels. Keep in sync with SolverGPU.writeParams().
 */
export const PARAMS_WGSL = /* wgsl */ `
struct Params {
  nx: u32, ny: u32, nz: u32, n: u32,
  tau0: f32, csmag2: f32, uin: f32, rhoIn: f32,
  ground: u32, strideX: u32, spongeStart: f32, spongeNu: f32,
  uGround: f32, emaAlpha: f32, uRef: f32, refX: u32,
  sideMode: u32, collision: u32, tauWall: f32, pad2: u32,
  spongeIn: f32, contactH: f32, pad4: f32, pad5: f32,
};
`;

export const DIRNAMES = ['x', 'y', 'z'];

/**
 * Fused pull-streaming + collision kernel.
 *
 * Boundary handling on the pulled link x ← x − cᵢ:
 *  - x−cᵢ left of the inlet       → equilibrium(ρ_in, U)
 *  - x−cᵢ right of the outlet     → zero gradient (read the population from the last column)
 *  - x−cᵢ below the ground        → specular (free slip) | bounce-back | moving-wall bounce-back
 *  - x−cᵢ above the roof / sides  → specular reflection (free slip)
 *  - x−cᵢ inside the vehicle      → halfway bounce-back
 */
export function streamCollideWGSL(L: Lattice, wg: number): string {
  const { c, w, q, opp, mirrorY, mirrorZ, mirrorYZ, dim } = L;
  const pairs = tensorPairs(dim);
  const comp = ['x', 'y', 'z'];
  let pull = '';
  for (let i = 0; i < q; i++) {
    const [cx, cy, cz] = c[i];
    if (i === 0) {
      pull += `  let f0 = fin[idx];\n`;
      continue;
    }
    const cu = [cx !== 0 ? `${cx}.0 * U` : '0.0'].join('');
    const eqIn = `${fl(w[i])} * P.rhoIn * (1.0 + 3.0 * (${cu}) + 4.5 * (${cu}) * (${cu}) - 1.5 * U * U)`;
    const movTerm = cx !== 0 ? ` + ${fl(6 * w[i] * cx)} * P.uGround` : '';
    let s = `  var f${i}: f32;\n  {\n`;
    s += `    var sx = ix${cx > 0 ? ' - 1' : cx < 0 ? ' + 1' : ''};\n`;
    s += `    var sy = iy${cy > 0 ? ' - 1' : cy < 0 ? ' + 1' : ''};\n`;
    s += dim === 3 ? `    var sz = iz${cz > 0 ? ' - 1' : cz < 0 ? ' + 1' : ''};\n` : `    let sz = 0;\n`;
    // Conditions (known statically which side may leave the domain)
    const inl = cx > 0 ? `sx < 0` : 'false';
    if (cx < 0) s += `    sx = min(sx, NX - 1);\n`;
    const yb = cy > 0 ? `sy < 0` : cy < 0 ? `sy >= NY` : 'false';
    const zb = cz > 0 ? `sz < 0` : cz < 0 ? `sz >= NZ` : 'false';
    s += `    let yOut = ${yb};\n    let zOut = ${zb};\n`;
    s += `    if (${inl}) {\n      f${i} = ${eqIn};\n    }`;
    if (cy > 0) {
      // ground (below): bounce back when not free-slip
      s += ` else if (yOut && P.ground != 0u) {\n      wall = true;\n      gwall = true;\n      f${i} = fin[${opp[i]}u * N + idx]${movTerm ? ` + belt * (${movTerm.slice(3)})` : ''};\n    }`;
    }
    s += ` else {\n`;
    s += `      let sxc = clamp(sx, 0, NX - 1);\n`;
    if (dim === 3) {
      s += `      if (yOut && zOut) {\n        f${i} = fin[${mirrorYZ[i]}u * N + cellId(sxc, iy, iz)];\n      } else `;
    }
    s += `      if (yOut) {\n        f${i} = fin[${mirrorY[i]}u * N + cellId(sxc, iy, ${dim === 3 ? 'clamp(sz, 0, NZ - 1)' : '0'})];\n      }`;
    if (dim === 3) {
      s += ` else if (zOut) {\n        if (P.sideMode == 1u) {\n          f${i} = fin[${i}u * N + cellId(sxc, sy, (sz + NZ) % NZ)];\n        } else {\n          f${i} = fin[${mirrorZ[i]}u * N + cellId(sxc, sy, iz)];\n        }\n      }`;
    }
    s += ` else {\n        let sc = cellId(sxc, sy, sz);\n        if (flags[sc] != 0u) { wall = true; f${i} = fin[${opp[i]}u * N + idx]; } else { f${i} = fin[${i}u * N + sc]; }\n      }\n`;
    s += `    }\n  }\n`;
    pull += s;
  }

  const fs = Array.from({ length: q }, (_, i) => `f${i}`);
  const sumExpr = (coef: (i: number) => number) => {
    const terms: string[] = [];
    for (let i = 0; i < q; i++) {
      const k = coef(i);
      if (k === 0) continue;
      terms.push(k === 1 ? fs[i] : k === -1 ? `-${fs[i]}` : `${fl(k)} * ${fs[i]}`);
    }
    return terms.length ? terms.join(' + ').replace(/\+ -/g, '- ') : '0.0';
  };
  const rho = sumExpr(() => 1);
  const mom = [0, 1, 2].slice(0, dim).map((a) => sumExpr((i) => c[i][a]));
  const pis = pairs.map(([a, b]) => sumExpr((i) => c[i][a] * c[i][b]));

  const uvars = comp.slice(0, dim).map((d) => `u${d}`);
  const usq = uvars.map((u) => `${u} * ${u}`).join(' + ');
  const piVars = pairs.map(([a, b]) => `p${comp[a]}${comp[b]}`);
  let collide = '';
  collide += `  let rho = ${rho};\n  let irho = 1.0 / rho;\n`;
  uvars.forEach((u, a) => (collide += `  let ${u} = (${mom[a]}) * irho;\n`));
  collide += `  let usq = ${usq};\n`;
  // Π^neq = Σ cc f − ρ(uu + δ/3)
  pairs.forEach(([a, b], k) => {
    const eq = a === b ? `rho * (${uvars[a]} * ${uvars[b]} + ${fl(1 / 3)})` : `rho * ${uvars[a]} * ${uvars[b]}`;
    collide += `  let ${piVars[k]} = (${pis[k]}) - ${eq};\n`;
  });
  const piNorm = pairs
    .map(([a, b], k) => (a === b ? `${piVars[k]} * ${piVars[k]}` : `2.0 * ${piVars[k]} * ${piVars[k]}`))
    .join(' + ');
  const trace = pairs.map(([a, b], k) => (a === b ? piVars[k] : '')).filter(Boolean).join(' + ');
  collide += `  let piN = sqrt(${piNorm});\n`;
  collide += `  let fx = f32(ix);\n`;
  collide += `  let sp = clamp((fx - P.spongeStart) / max(f32(NX) - P.spongeStart, 1.0), 0.0, 1.0);\n`;
  collide += `  var t0 = P.tau0 + 3.0 * P.spongeNu * sp * sp;\n`;
  // viscosity floor on floor-adjacent cells only (damps the moving-belt wall mode). Applying it on the
  // body would add a spurious Couette shear drag through the under-body gap.
  collide += `  if (gwall) { t0 = max(t0, P.tauWall); }\n`;
  collide += `  let tau = 0.5 * (t0 + sqrt(t0 * t0 + ${fl(18 * Math.SQRT2)} * P.csmag2 * piN * irho));\n`;
  collide += `  let k1 = (1.0 - 1.0 / tau) * 4.5;\n`;
  // hybrid: plain BGK on wall-adjacent cells (the regularized projection excites an odd-even wall
  // mode with bounce-back), regularized BGK in the bulk where its stability at tau→1/2 matters.
  collide += `  let useBGK = wall || P.collision == 1u;\n`;
  // absorbing layer at the inlet: damps pressure waves reflected back upstream
  collide += `  let si = clamp(1.0 - fx / max(P.spongeIn, 1.0), 0.0, 1.0);\n  let sIn = select(0.0, 0.08 * si * si, P.spongeIn > 0.0);\n`;
  collide += `  let trP = (${trace}) * ${fl(1 / 3)};\n`;
  for (let i = 0; i < q; i++) {
    const [cxi, cyi, czi] = c[i];
    const cv = [cxi, cyi, czi];
    const cuTerms = uvars.map((u, a) => (cv[a] === 0 ? '' : `${cv[a] === 1 ? '' : '-'}${u}`)).filter(Boolean);
    const cu = cuTerms.length ? cuTerms.join(' + ').replace(/\+ -/g, '- ') : '0.0';
    // Qᵢ:Π = Σ cᵢₐ cᵢᵦ Πₐᵦ − tr(Π)/3
    const qp: string[] = [];
    pairs.forEach(([a, b], k) => {
      const coef = cv[a] * cv[b] * (a === b ? 1 : 2);
      if (coef !== 0) qp.push(`${fl(coef)} * ${piVars[k]}`);
    });
    const qpi = (qp.length ? qp.join(' + ') : '0.0') + ' - trP';
    collide += `  {\n    let cu = ${cu};\n    let feq = ${fl(w[i])} * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);\n`;
    collide += `    let fr${i} = ${fl(w[i])} * k1 * (${qpi});\n`;
    collide += `    var fo${i} = feq + select(fr${i}, (1.0 - 1.0 / tau) * (f${i} - feq), useBGK);\n`;
    const cxi2 = c[i][0];
    collide += `    if (sIn > 0.0) { fo${i} = mix(fo${i}, ${fl(w[i])} * (1.0 + ${fl(3 * cxi2)} * U + ${fl(4.5 * cxi2 * cxi2)} * U * U - 1.5 * U * U), sIn); }\n`;
    collide += `    fout[${i}u * N + idx] = fo${i};\n  }\n`;
  }

  return /* wgsl */ `
${PARAMS_WGSL}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> fin: array<f32>;
@group(0) @binding(2) var<storage, read_write> fout: array<f32>;
@group(0) @binding(3) var<storage, read> flags: array<u32>;

var<private> NX: i32;
var<private> NY: i32;
var<private> NZ: i32;

fn cellId(x: i32, y: i32, z: i32) -> u32 {
  return u32(x + NX * (y + NY * z));
}

@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x + gid.y * P.strideX;
  let N = P.n;
  if (idx >= N) { return; }
  NX = i32(P.nx); NY = i32(P.ny); NZ = i32(P.nz);
  let ix = i32(idx % P.nx);
  let iy = i32((idx / P.nx) % P.ny);
  let iz = i32(idx / (P.nx * P.ny));
  let U = P.uin;
  var wall = false;
  var gwall = false;
  // tyre contact patches: the belt is held still within a small radius of the body, otherwise it
  // drags the stagnant fluid of the tyre/belt wedge straight into the (non-rotating) tyres
  var belt = select(0.0, 1.0, P.ground == 2u);
  if (belt > 0.0 && iy == 0) {
    let r = i32(P.contactH);
    for (var dz = -r; dz <= r; dz++) {
      for (var dx = -r; dx <= r; dx++) {
        for (var dy = 0; dy <= 1; dy++) {
          let qx = ix + dx; let qz = iz + dz;
          if (qx >= 0 && qx < NX && qz >= 0 && qz < NZ && flags[cellId(qx, dy, qz)] != 0u) { belt = 0.0; }
        }
      }
    }
  }
  if (flags[idx] != 0u) {
${Array.from({ length: q }, (_, i) => `    fout[${i}u * N + idx] = ${fl(w[i])};`).join('\n')}
    return;
  }
${pull}
${collide}
}
`;
}

/** Initialise all populations to equilibrium at (ρ=1, u=(u0,0,0)). */
export function initWGSL(L: Lattice, wg: number): string {
  const { c, w, q } = L;
  let body = '';
  for (let i = 0; i < q; i++) {
    const cx = c[i][0];
    body += `  fout[${i}u * N + idx] = ${fl(w[i])} * (1.0 + ${fl(3 * cx)} * U + ${fl(4.5 * cx * cx)} * U * U - 1.5 * U * U);\n`;
  }
  return /* wgsl */ `
${PARAMS_WGSL}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(2) var<storage, read_write> fout: array<f32>;
@group(0) @binding(3) var<storage, read> flags: array<u32>;
@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x + gid.y * P.strideX;
  let N = P.n;
  if (idx >= N) { return; }
  var U = P.uin;
  if (flags[idx] != 0u) { U = 0.0; }
${body}
}
`;
}

/**
 * Macroscopic pass: ρ, u → vel texture, EMA statistics, momentum-exchange force on vehicle links,
 * and per-workgroup partial reductions (Fx, Fy, Fz, max|u|²) and (Σρ_ref, n_ref, nan, solidCount).
 */
export function macroWGSL(L: Lattice, wg: number): string {
  const { c, q, dim } = L;
  const fs = Array.from({ length: q }, (_, i) => `f[${i}]`);
  const sum = (coef: (i: number) => number) =>
    fs
      .map((f, i) => {
        const k = coef(i);
        return k === 0 ? '' : k === 1 ? `+ ${f}` : k === -1 ? `- ${f}` : `+ ${fl(k)} * ${f}`;
      })
      .filter(Boolean)
      .join(' ')
      .replace(/^\+ /, '')
      .replace(/^- /, '-');
  let loads = '';
  for (let i = 0; i < q; i++) loads += `  f[${i}] = fin[${i}u * N + idx];\n`;
  // Momentum exchange on links pointing into the vehicle: F += 2 (f_i − w_i ρ₀) c_i. Subtracting the
  // reference state removes the absolute pressure ρ₀c_s² acting on surfaces that touch the floor.
  let force = '';
  for (let i = 1; i < q; i++) {
    const [cx, cy, cz] = c[i];
    const nb = `cellIdSafe(ix + ${cx}, iy + ${cy}, iz + ${cz})`;
    const fv = [cx, cy, cz]
      .map((v, a) => (v === 0 ? '' : `F.${'xyz'[a]} += ${fl(2 * v)} * (f[${i}] - ${fl(L.w[i])});`))
      .join(' ');
    force += `    if (isSolid(${nb})) { ${fv} }\n`;
  }
  let solidRho = '';
  const nbs = dim === 3 ? [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] : [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]];
  for (const [a, b, d] of nbs) {
    solidRho += `    { let s = cellIdSafe(ix + ${a}, iy + ${b}, iz + ${d}); if (s != 0xffffffffu && flags[s] == 0u) { var r = 0.0; for (var k = 0u; k < ${q}u; k++) { r += fin[k * N + s]; } rs += r; rn += 1.0; } }\n`;
  }
  return /* wgsl */ `
${PARAMS_WGSL}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> fin: array<f32>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var velTex: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var<storage, read_write> meanA: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> meanB: array<f32>;
@group(0) @binding(6) var<storage, read_write> partials: array<vec4<f32>>;

var<private> NX: i32;
var<private> NY: i32;
var<private> NZ: i32;
var<workgroup> shA: array<vec4<f32>, ${wg}>;
var<workgroup> shB: array<vec4<f32>, ${wg}>;

fn cellIdSafe(x: i32, y: i32, z: i32) -> u32 {
  if (x < 0 || y < 0 || z < 0 || x >= NX || y >= NY || z >= NZ) { return 0xffffffffu; }
  return u32(x + NX * (y + NY * z));
}
fn isSolid(s: u32) -> bool {
  return s != 0xffffffffu && flags[s] != 0u;
}

@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32,
        @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = gid.x + gid.y * P.strideX;
  let N = P.n;
  NX = i32(P.nx); NY = i32(P.ny); NZ = i32(P.nz);
  var A = vec4<f32>(0.0);
  var B = vec4<f32>(0.0);
  if (idx < N) {
    let ix = i32(idx % P.nx);
    let iy = i32((idx / P.nx) % P.ny);
    let iz = i32(idx / (P.nx * P.ny));
    var out = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if (flags[idx] == 0u) {
      var f: array<f32, ${q}>;
${loads}
      let rho = ${sum(() => 1)};
      let u = vec3<f32>(${[0, 1, 2].map((a) => (a < dim ? `(${sum((i) => c[i][a])}) / rho` : '0.0')).join(', ')});
      out = vec4<f32>(u, rho);
      var F = vec3<f32>(0.0);
${force}
      let uu = dot(u, u);
      let bad = select(0.0, 1.0, !(uu < 1.0e6) || !(rho > 0.0));
      A = vec4<f32>(F, select(uu, 1.0e6, bad > 0.0));
      if (u32(ix) == P.refX) { B = vec4<f32>(rho, 1.0, bad, 0.0); } else { B = vec4<f32>(0.0, 0.0, bad, 0.0); }
      let m = meanA[idx];
      let a = P.emaAlpha;
      meanA[idx] = mix(m, out, a);
      meanB[idx] = mix(meanB[idx], uu, a);
    } else {
      // extrapolate density into the solid so trilinear sampling at the wall is meaningful
      var rs = 0.0; var rn = 0.0;
${solidRho}
      if (rn > 0.0) { out.w = rs / rn; }
      meanA[idx] = mix(meanA[idx], out, P.emaAlpha);
      meanB[idx] = 0.0;
      B = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    textureStore(velTex, vec3<i32>(ix, iy, iz), out);
  }
  shA[li] = A;
  shB[li] = B;
  workgroupBarrier();
  for (var s = ${wg / 2}u; s > 0u; s = s >> 1u) {
    if (li < s) {
      let a2 = shA[li + s];
      let b2 = shB[li + s];
      shA[li] = vec4<f32>(shA[li].xyz + a2.xyz, max(shA[li].w, a2.w));
      shB[li] = shB[li] + b2;
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    let wi = wid.x + wid.y * nwg.x;
    partials[2u * wi] = shA[0];
    partials[2u * wi + 1u] = shB[0];
  }
}
`;
}

export function reduceWGSL(): string {
  return /* wgsl */ `
struct RP { count: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> R: RP;
@group(0) @binding(1) var<storage, read> partials: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> result: array<vec4<f32>>;
var<workgroup> shA: array<vec4<f32>, 256>;
var<workgroup> shB: array<vec4<f32>, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  var A = vec4<f32>(0.0);
  var B = vec4<f32>(0.0);
  for (var i = li; i < R.count; i += 256u) {
    let a = partials[2u * i];
    A = vec4<f32>(A.xyz + a.xyz, max(A.w, a.w));
    B += partials[2u * i + 1u];
  }
  shA[li] = A; shB[li] = B;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (li < s) {
      let a2 = shA[li + s];
      shA[li] = vec4<f32>(shA[li].xyz + a2.xyz, max(shA[li].w, a2.w));
      shB[li] = shB[li] + shB[li + s];
    }
    workgroupBarrier();
  }
  if (li == 0u) { result[0] = shA[0]; result[1] = shB[0]; }
}
`;
}

/**
 * Derived fields: vorticity + Q-criterion from central differences of the velocity texture,
 * turbulence intensity from the EMA statistics.
 */
export function derivedWGSL(dim: 2 | 3): string {
  return /* wgsl */ `
${PARAMS_WGSL}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var velIn: texture_3d<f32>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read> meanA: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> meanB: array<f32>;
@group(0) @binding(5) var vortTex: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var statTex: texture_storage_3d<rgba16float, write>;
@group(0) @binding(7) var meanTex: texture_storage_3d<rgba16float, write>;

fn isSolidAt(q: vec3<i32>) -> bool {
  if (any(q < vec3<i32>(0)) || q.x >= i32(P.nx) || q.y >= i32(P.ny) || q.z >= i32(P.nz)) { return false; }
  return flags[u32(q.x) + P.nx * (u32(q.y) + P.ny * u32(q.z))] != 0u;
}
fn vel(p: vec3<i32>) -> vec3<f32> {
  let q = clamp(p, vec3<i32>(0), vec3<i32>(i32(P.nx) - 1, i32(P.ny) - 1, i32(P.nz) - 1));
  return textureLoad(velIn, q, 0).xyz;
}

@compute @workgroup_size(8, 8, ${dim === 3 ? 4 : 1})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.nx || gid.y >= P.ny || gid.z >= P.nz) { return; }
  let p = vec3<i32>(gid);
  let idx = gid.x + P.nx * (gid.y + P.ny * gid.z);
  let solid = flags[idx] != 0u;
  var vort = vec4<f32>(0.0);
  if (!solid) {
    let dx = 0.5 * (vel(p + vec3<i32>(1, 0, 0)) - vel(p - vec3<i32>(1, 0, 0)));
    let dy = 0.5 * (vel(p + vec3<i32>(0, 1, 0)) - vel(p - vec3<i32>(0, 1, 0)));
    ${dim === 3 ? 'let dz = 0.5 * (vel(p + vec3<i32>(0, 0, 1)) - vel(p - vec3<i32>(0, 0, 1)));' : 'let dz = vec3<f32>(0.0);'}
    // J[i][j] = du_i/dx_j ; columns dx,dy,dz
    let J = mat3x3<f32>(dx, dy, dz); // J[col][row]: J[j][i] = du_i/dx_j
    let wx = dy.z - dz.y;
    let wy = dz.x - dx.z;
    let wz = dx.y - dy.x;
    var S2 = 0.0; var W2 = 0.0;
    for (var i = 0; i < 3; i++) {
      for (var j = 0; j < 3; j++) {
        let s = 0.5 * (J[j][i] + J[i][j]);
        let w = 0.5 * (J[j][i] - J[i][j]);
        S2 += s * s; W2 += w * w;
      }
    }
    // Q is dominated by staircase shear right at the wall; fade it out within two cells of the body
    // so iso-surfaces show the separated vortices rather than coating the vehicle
    var near = 0.0;
    for (var d = 1; d <= 2; d++) {
      for (var a = 0; a < ${dim}; a++) {
        var e = vec3<i32>(0);
        e[a] = d;
        if (isSolidAt(p + e) || isSolidAt(p - e)) { near = max(near, select(0.5, 1.0, d == 1)); }
      }
    }
    vort = vec4<f32>(wx, wy, wz, 0.5 * (W2 - S2) * (1.0 - near));
  }
  let m = meanA[idx];
  let k = max(meanB[idx] - dot(m.xyz, m.xyz), 0.0);
  let ti = sqrt(k / 3.0) / max(P.uRef, 1e-6);
  textureStore(vortTex, p, vort);
  textureStore(statTex, p, vec4<f32>(select(ti, 0.0, solid), select(0.0, 1.0, solid), 0.0, 0.0));
  textureStore(meanTex, p, m);
}
`;
}
