import { FRAME_WGSL } from './wgslRender';

export const PARTICLES_WGSL = /* wgsl */ `
${FRAME_WGSL}
struct PU {
  count: u32, trail: u32, head: u32, emitter: u32,
  steps: f32, seed: u32, maxAge: f32, nozzles: u32,
  rakeA: vec4<f32>,
  rakeB: vec4<f32>,
  advance: u32, fill: u32, p0: u32, p1: u32,
};
@group(1) @binding(0) var<uniform> P: PU;
@group(1) @binding(1) var<storage, read_write> state: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> hist: array<vec4<f32>>;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
var<private> rngState: u32;
fn rnd() -> f32 {
  rngState = pcg(rngState);
  return f32(rngState) / 4294967295.0;
}

fn spawn() -> vec3<f32> {
  let d = F.dims.xyz;
  let is2D = F.dims.w > 0.5;
  if (P.emitter == 0u) {
    return vec3<f32>(1.0 + rnd() * 2.0, 0.5 + rnd() * (d.y - 1.0), select(0.5 + rnd() * (d.z - 1.0), 0.5, is2D));
  }
  let k = min(u32(rnd() * f32(P.nozzles)), P.nozzles - 1u);
  let t = (f32(k) + 0.5) / f32(P.nozzles);
  var p = mix(P.rakeA.xyz, P.rakeB.xyz, t);
  let j = vec3<f32>(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5) * 0.35;
  p += j;
  if (is2D) { p.z = 0.5; }
  return p;
}

fn fillPos() -> vec3<f32> {
  let d = F.dims.xyz;
  let is2D = F.dims.w > 0.5;
  for (var k = 0; k < 8; k++) {
    let p = vec3<f32>(1.0 + rnd() * (d.x - 2.0), 0.5 + rnd() * (d.y - 1.0), select(0.5 + rnd() * (d.z - 1.0), 0.5, is2D));
    if (!solidAt(p)) { return p; }
  }
  return spawn();
}

fn vel(p: vec3<f32>) -> vec3<f32> {
  var v = sVel(p).xyz;
  if (F.dims.w > 0.5) { v.z = 0.0; }
  return v;
}

fn outside(p: vec3<f32>) -> bool {
  let d = F.dims.xyz;
  if (F.dims.w > 0.5) { return p.x < 0.0 || p.y < 0.0 || p.x >= d.x - 1.0 || p.y >= d.y; }
  return any(p < vec3<f32>(0.0)) || p.x >= d.x - 1.0 || p.y >= d.y || p.z >= d.z;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.count) { return; }
  rngState = pcg(i ^ pcg(P.seed));
  var s = state[i];
  if (P.fill != 0u) {
    if (P.emitter == 0u) {
      s = vec4<f32>(fillPos(), f32(P.trail));
    } else {
      s = vec4<f32>(spawn(), -floor(rnd() * P.maxAge));
    }
    state[i] = s;
    for (var k = 0u; k < P.trail; k++) { hist[k * P.count + i] = vec4<f32>(s.xyz, 0.0); }
    return;
  }
  if (P.advance == 0u) { return; }
  if (s.w < 0.0) {
    s.w += 1.0;
    if (s.w >= 0.0) { s = vec4<f32>(spawn(), 0.0); }
  } else {
    var p = s.xyz;
    let v0 = vel(p);
    let disp = length(v0) * P.steps;
    let nsub = u32(clamp(ceil(disp / 0.6), 1.0, 24.0));
    let h = P.steps / f32(nsub);
    for (var k = 0u; k < nsub; k++) {
      let a = vel(p);
      let b = vel(p + a * (0.5 * h));
      p += b * h;
    }
    s = vec4<f32>(p, s.w + 1.0);
    if (outside(p) || solidAt(p) || s.w > P.maxAge * 4.0) {
      s = vec4<f32>(spawn(), 0.0);
    }
  }
  state[i] = s;
  hist[P.head * P.count + i] = vec4<f32>(s.xyz, 0.0);
}
`;

/** Samples the instantaneous and mean fields at up to 8 probe points. */
export const PROBES_WGSL = /* wgsl */ `
${FRAME_WGSL}
struct PR { count: u32, p0: u32, p1: u32, p2: u32, pos: array<vec4<f32>, 8> };
@group(1) @binding(0) var<uniform> PRB: PR;
@group(1) @binding(1) var<storage, read_write> probeOut: array<vec4<f32>>;
@compute @workgroup_size(8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= PRB.count) { return; }
  let p = PRB.pos[i].xyz;
  var v = sVel(p);
  var m = sMean(p);
  if (F.dims.w > 0.5) { v.z = 0.0; m.z = 0.0; }
  probeOut[2u * i] = v;
  probeOut[2u * i + 1u] = m;
}
`;

/** Brick pyramid for empty-space skipping in the iso-surface ray-march. */
export const BRICKS_WGSL = /* wgsl */ `
${FRAME_WGSL}
@group(1) @binding(0) var brickOut: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let bd = textureDimensions(brickOut);
  if (any(gid >= bd)) { return; }
  let d = vec3<i32>(F.dims.xyz) - vec3<i32>(1);
  var mq = -1e9;
  var mu = 1e9;
  let o = vec3<i32>(gid) * 4;
  for (var z = -1; z <= 4; z++) {
    for (var y = -1; y <= 4; y++) {
      for (var x = -1; x <= 4; x++) {
        let c = clamp(o + vec3<i32>(x, y, z), vec3<i32>(0), d);
        mq = max(mq, textureLoad(vortTex, c, 0).w);
        if (textureLoad(statTex, c, 0).y < 0.5) { mu = min(mu, textureLoad(meanTex, c, 0).x); }
      }
    }
  }
  textureStore(brickOut, vec3<i32>(gid), vec4<f32>(mq, mu, 0.0, 0.0));
}
`;

export const STREAMLINES_WGSL = /* wgsl */ `
${FRAME_WGSL}
struct SLU {
  seeds: u32, points: u32, useMean: u32, bidir: u32,
  rakeA: vec4<f32>,
  rakeB: vec4<f32>,
  h: f32, p0: f32, p1: f32, p2: f32,
};
@group(1) @binding(0) var<uniform> S: SLU;
@group(1) @binding(1) var<storage, read_write> pts: array<vec4<f32>>;

fn vel(p: vec3<f32>) -> vec3<f32> {
  var v = sVel(p).xyz;
  if (S.useMean != 0u) { v = sMean(p).xyz; }
  if (F.dims.w > 0.5) { v.z = 0.0; }
  return v;
}
fn outside(p: vec3<f32>) -> bool {
  let d = F.dims.xyz;
  if (F.dims.w > 0.5) { return p.x < 0.0 || p.y < 0.0 || p.x >= d.x || p.y >= d.y; }
  return any(p < vec3<f32>(0.0)) || any(p >= d);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  if (s >= S.seeds) { return; }
  var p = mix(S.rakeA.xyz, S.rakeB.xyz, (f32(s) + 0.5) / f32(S.seeds));
  if (F.dims.w > 0.5) { p.z = 0.5; }
  var alive = true;
  let U = max(F.flow.x, 1e-5);
  for (var k = 0u; k < S.points; k++) {
    let v = vel(p);
    let sp = length(v);
    if (alive && (outside(p) || solidAt(p) || sp < U * 1e-3)) { alive = false; }
    pts[s * S.points + k] = vec4<f32>(p, select(-1.0, sp, alive));
    if (!alive) { continue; }
    let a = v / sp;
    let vm = vel(p + a * (0.5 * S.h));
    let lm = length(vm);
    if (lm < 1e-9) { alive = false; continue; }
    p += vm / lm * S.h;
  }
}
`;
