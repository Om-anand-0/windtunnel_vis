import { COLORMAP_WGSL } from './colormaps';

/** Group 0 shared by every render/visualization pipeline. */
export const FRAME_WGSL = /* wgsl */ `
struct Frame {
  viewProj: mat4x4<f32>,
  invViewProj: mat4x4<f32>,
  eye: vec4<f32>,
  viewport: vec4<f32>,   // w, h, 1/w, 1/h
  dims: vec4<f32>,       // nx, ny, nz, is2D
  flow: vec4<f32>,       // U (lattice), rhoRef, Lref (cells), time
  extra: vec4<f32>,      // groundOffset, dpr, 0, 0
};
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var velTex: texture_3d<f32>;
@group(0) @binding(2) var vortTex: texture_3d<f32>;
@group(0) @binding(3) var statTex: texture_3d<f32>;
@group(0) @binding(4) var samp: sampler;
@group(0) @binding(5) var<storage, read> flagsBuf: array<u32>;
@group(0) @binding(6) var meanTex: texture_3d<f32>;

fn uvw(p: vec3<f32>) -> vec3<f32> {
  var t = p / F.dims.xyz;
  if (F.dims.w > 0.5) { t.z = 0.5; }
  return t;
}
fn sVel(p: vec3<f32>) -> vec4<f32> { return textureSampleLevel(velTex, samp, uvw(p), 0.0); }
fn sVort(p: vec3<f32>) -> vec4<f32> { return textureSampleLevel(vortTex, samp, uvw(p), 0.0); }
fn sStat(p: vec3<f32>) -> vec4<f32> { return textureSampleLevel(statTex, samp, uvw(p), 0.0); }
fn sMean(p: vec3<f32>) -> vec4<f32> { return textureSampleLevel(meanTex, samp, uvw(p), 0.0); }
fn solidAt(p: vec3<f32>) -> bool {
  let d = vec3<i32>(F.dims.xyz);
  var c = vec3<i32>(floor(p));
  if (F.dims.w > 0.5) { c.z = 0; }
  if (any(c < vec3<i32>(0)) || any(c >= d)) { return false; }
  return flagsBuf[u32(c.x + d.x * (c.y + d.y * c.z))] != 0u;
}
fn cpOf(rho: f32) -> f32 {
  let U = max(F.flow.x, 1e-5);
  return 2.0 * (rho - F.flow.y) / (3.0 * U * U);
}
${COLORMAP_WGSL}
`;

export const MESH_WGSL = /* wgsl */ `
${FRAME_WGSL}
struct MeshU {
  model: mat4x4<f32>,
  normalM: mat4x4<f32>,
  mode: u32, cmap: u32, useMean: u32, flat2D: u32,
  cpMin: f32, cpMax: f32, opacity: f32, pad: f32,
};
@group(1) @binding(0) var<uniform> M: MeshU;

struct VO {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) n: vec3<f32>,
};

@vertex fn vs(@location(0) p: vec3<f32>, @location(1) n: vec3<f32>) -> VO {
  var o: VO;
  let w = M.model * vec4<f32>(p, 1.0);
  o.world = w.xyz;
  o.n = normalize((M.normalM * vec4<f32>(n, 0.0)).xyz);
  o.pos = F.viewProj * w;
  return o;
}

fn surfaceCp(world: vec3<f32>, n: vec3<f32>) -> f32 {
  var nn = n;
  var base = world;
  if (M.flat2D != 0u) {
    base.z = 0.5;
    nn = vec3<f32>(n.xy, 0.0);
    let l = length(nn);
    nn = select(vec3<f32>(0.0, 1.0, 0.0), nn / l, l > 1e-3);
  }
  var q = base + nn * 1.5;
  if (solidAt(q)) { q = base - nn * 1.5; }
  if (solidAt(q)) { q = base + nn * 2.5; }
  var rho = sVel(q).w;
  if (M.useMean != 0u) { rho = sMean(q).w; }
  return cpOf(rho);
}

@fragment fn fs(i: VO, @builtin(front_facing) ff: bool) -> @location(0) vec4<f32> {
  var n = normalize(i.n);
  if (!ff) { n = -n; }
  let v = normalize(F.eye.xyz - i.world);
  let key = normalize(vec3<f32>(-0.45, 0.8, 0.55));
  let fill = normalize(vec3<f32>(0.6, 0.3, -0.7));
  let lam = max(dot(n, key), 0.0);
  let lam2 = max(dot(n, fill), 0.0);
  let hemi = 0.5 + 0.5 * n.y;
  let fres = pow(1.0 - max(dot(n, v), 0.0), 4.0);
  var col: vec3<f32>;
  if (M.mode == 1u) {
    let cp = surfaceCp(i.world, n);
    let t = (cp - M.cpMin) / (M.cpMax - M.cpMin);
    let c = colormap(M.cmap, t);
    col = c * (0.62 + 0.3 * lam + 0.12 * hemi) + vec3<f32>(0.15) * fres;
  } else {
    let base = vec3<f32>(0.46, 0.5, 0.56);
    let h = normalize(key + v);
    let spec = pow(max(dot(n, h), 0.0), 60.0) * 0.6;
    col = base * (0.18 + 0.62 * lam + 0.18 * lam2 + 0.22 * hemi) + vec3<f32>(spec) + vec3<f32>(0.35, 0.45, 0.6) * fres * 0.6;
  }
  return vec4<f32>(col * M.opacity, M.opacity);
}
`;

export const SLICE_WGSL = /* wgsl */ `
${FRAME_WGSL}
struct SliceU {
  axis: u32, field: u32, cmap: u32, flags: u32,
  pos: f32, vmin: f32, vmax: f32, opacity: f32,
  qThr: f32, p0: f32, p1: f32, p2: f32,
};
@group(1) @binding(0) var<uniform> S: SliceU;

struct VO {
  @builtin(position) pos: vec4<f32>,
  @location(0) gp: vec3<f32>,
};

fn planePoint(uv: vec2<f32>) -> vec3<f32> {
  let d = F.dims.xyz;
  if (S.axis == 0u) { return vec3<f32>(S.pos, uv.x * d.y, uv.y * d.z); }
  if (S.axis == 1u) { return vec3<f32>(uv.x * d.x, S.pos, uv.y * d.z); }
  return vec3<f32>(uv.x * d.x, uv.y * d.y, S.pos);
}

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VO {
  var q = array<vec2<f32>, 6>(vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));
  var o: VO;
  o.gp = planePoint(q[vi]);
  o.pos = F.viewProj * vec4<f32>(o.gp, 1.0);
  return o;
}

@fragment fn fs(i: VO) -> @location(0) vec4<f32> {
  let p = i.gp;
  let vel = sVel(p);
  let st = sStat(p);
  let mn = sMean(p);
  let vo = sVort(p);
  let U = max(F.flow.x, 1e-5);
  let L = F.flow.z;
  let useMean = (S.flags & 1u) != 0u;
  var v = 0.0;
  switch (S.field) {
    case 0u: { v = select(length(vel.xyz), length(mn.xyz), useMean) / U; }
    case 1u: { v = select(vel.x, mn.x, useMean) / U; }
    case 2u: { v = cpOf(select(vel.w, mn.w, useMean)); }
    case 3u: {
      var w = vo.z;
      if (S.axis == 0u) { w = vo.x; } else if (S.axis == 1u) { w = vo.y; }
      v = w * L / U;
    }
    case 4u: { v = st.x; }
    default: { v = vo.w * L * L / (U * U); }
  }
  // iso contours (derivatives evaluated in uniform control flow)
  let rc = mn.x / U;
  let rcw = fwidth(rc);
  let qn = vo.w * L * L / (U * U) - S.qThr;
  let qw = fwidth(qn);
  let sw = fwidth(st.y);
  var col = colormap(S.cmap, (v - S.vmin) / (S.vmax - S.vmin));
  if ((S.flags & 2u) != 0u) {
    if (rc < 0.0) { col = mix(col, vec3<f32>(1.0, 1.0, 1.0), 0.18); }
    let e = abs(rc) / max(rcw, 1e-6);
    col = mix(vec3<f32>(1.0, 1.0, 1.0), col, smoothstep(0.5, 1.5, e));
  }
  if ((S.flags & 4u) != 0u) {
    let e = abs(qn) / max(qw, 1e-6);
    col = mix(vec3<f32>(0.05, 0.05, 0.05), col, smoothstep(0.5, 1.5, e));
  }
  // solid body: flat dark fill with a crisp outline
  let edge = smoothstep(0.35, 0.65, st.y);
  let outline = 1.0 - smoothstep(0.0, 1.5, abs(st.y - 0.5) / max(sw, 1e-4));
  col = mix(col, vec3<f32>(0.11, 0.12, 0.14), edge);
  col = mix(col, vec3<f32>(0.85, 0.88, 0.92), outline * 0.8);
  return vec4<f32>(col * S.opacity, S.opacity);
}
`;

export const VOLUME_WGSL = /* wgsl */ `
${FRAME_WGSL}
struct VolU {
  thr: f32, mode: u32, colorBy: u32, maxSteps: u32,
  stepLen: f32, opacity: f32, p0: f32, p1: f32,
};
@group(1) @binding(0) var<uniform> V: VolU;
// 4³-cell bricks: (max Q, min mean uₓ) incl. a 1-cell border — rays skip bricks that cannot hold the surface
@group(1) @binding(1) var brickTex: texture_3d<f32>;

fn skipBrick(b: vec4<f32>) -> bool {
  let U = max(F.flow.x, 1e-5);
  let L = F.flow.z;
  if (V.mode == 0u) { return b.x * L * L / (U * U) - V.thr < 0.0; }
  return -b.y / U - V.thr * 0.02 < 0.0;
}

struct VO { @builtin(position) pos: vec4<f32>, @location(0) ndc: vec2<f32> };

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VO {
  var q = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  var o: VO;
  o.pos = vec4<f32>(q[vi], 0.0, 1.0);
  o.ndc = q[vi];
  return o;
}

fn field(p: vec3<f32>) -> f32 {
  let U = max(F.flow.x, 1e-5);
  let L = F.flow.z;
  if (V.mode == 0u) {
    return sVort(p).w * L * L / (U * U) - V.thr;
  }
  return -sMean(p).x / U - V.thr * 0.02 - sStat(p).y * 10.0;
}

fn hash(p: vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

struct FO { @location(0) color: vec4<f32>, @builtin(frag_depth) depth: f32 };

@fragment fn fs(i: VO) -> FO {
  let a = F.invViewProj * vec4<f32>(i.ndc, 0.0, 1.0);
  let b = F.invViewProj * vec4<f32>(i.ndc, 1.0, 1.0);
  let ro = a.xyz / a.w;
  let rd = normalize(b.xyz / b.w - ro);
  let lo = vec3<f32>(1.0);
  let hi = F.dims.xyz - vec3<f32>(1.0);
  let inv = 1.0 / rd;
  let t0 = (lo - ro) * inv;
  let t1 = (hi - ro) * inv;
  let tmin = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
  let tmax = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
  var o: FO;
  if (tmax <= max(tmin, 0.0)) { discard; }
  var t = max(tmin, 0.0) + V.stepLen * hash(i.pos.xy);
  var prev = field(ro + rd * t);
  var hit = false;
  var th = 0.0;
  let bd = vec3<i32>(textureDimensions(brickTex)) - vec3<i32>(1);
  let rdi = 1.0 / (rd + vec3<f32>(1e-12));
  for (var k = 0u; k < V.maxSteps * 2u; k++) {
    let bi = clamp(vec3<i32>(floor((ro + rd * t) / 4.0)), vec3<i32>(0), bd);
    if (skipBrick(textureLoad(brickTex, bi, 0))) {
      let bmin = vec3<f32>(bi) * 4.0;
      let tb3 = (select(bmin, bmin + vec3<f32>(4.0), rd > vec3<f32>(0.0)) - ro) * rdi;
      t = max(min(min(tb3.x, tb3.y), tb3.z), t) + 0.02;
      prev = -1.0;
      if (t > tmax) { break; }
      continue;
    }
    let tn = t + V.stepLen;
    if (tn > tmax) { break; }
    let fv = field(ro + rd * tn);
    if (fv > 0.0 && prev <= 0.0) {
      var ta = t; var tb = tn;
      for (var j = 0; j < 5; j++) {
        let tm = 0.5 * (ta + tb);
        if (field(ro + rd * tm) > 0.0) { tb = tm; } else { ta = tm; }
      }
      th = tb;
      hit = true;
      break;
    }
    prev = fv;
    t = tn;
  }
  if (!hit) { discard; }
  let p = ro + rd * th;
  let e = 0.75;
  let g = vec3<f32>(
    field(p + vec3<f32>(e, 0.0, 0.0)) - field(p - vec3<f32>(e, 0.0, 0.0)),
    field(p + vec3<f32>(0.0, e, 0.0)) - field(p - vec3<f32>(0.0, e, 0.0)),
    field(p + vec3<f32>(0.0, 0.0, e)) - field(p - vec3<f32>(0.0, 0.0, e)));
  var n = -normalize(g + vec3<f32>(1e-7));
  if (dot(n, rd) > 0.0) { n = -n; }
  let U = max(F.flow.x, 1e-5);
  var base: vec3<f32>;
  if (V.colorBy == 1u) {
    let wx = sVort(p).x;
    base = select(vec3<f32>(0.25, 0.55, 1.0), vec3<f32>(1.0, 0.35, 0.25), wx > 0.0);
  } else if (V.mode == 1u) {
    base = vec3<f32>(0.55, 0.85, 1.0);
  } else {
    base = cm_turbo(length(sVel(p).xyz) / U * 0.75);
  }
  let key = normalize(vec3<f32>(-0.45, 0.8, 0.55));
  let lam = max(dot(n, key), 0.0);
  let rim = pow(1.0 - abs(dot(n, rd)), 3.0);
  let col = base * (0.3 + 0.7 * lam) + vec3<f32>(0.25) * rim;
  let clip = F.viewProj * vec4<f32>(p, 1.0);
  o.color = vec4<f32>(col * V.opacity, V.opacity);
  o.depth = clip.z / clip.w;
  return o;
}
`;

/** Screen-space ribbon helper shared by trails, streamlines and overlay segments. */
const RIBBON = /* wgsl */ `
struct RO {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) side: f32,
};
fn ribbon(p0: vec3<f32>, p1: vec3<f32>, corner: u32, widthPx: f32, color: vec4<f32>) -> RO {
  var o: RO;
  let c0 = F.viewProj * vec4<f32>(p0, 1.0);
  let c1 = F.viewProj * vec4<f32>(p1, 1.0);
  if (c0.w <= 0.01 || c1.w <= 0.01) {
    o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    o.color = vec4<f32>(0.0);
    return o;
  }
  let s0 = c0.xy / c0.w * F.viewport.xy * 0.5;
  let s1 = c1.xy / c1.w * F.viewport.xy * 0.5;
  var d = s1 - s0;
  let l = length(d);
  d = select(vec2<f32>(1.0, 0.0), d / l, l > 1e-5);
  let nrm = vec2<f32>(-d.y, d.x) * widthPx * 0.5;
  var cs = array<vec2<f32>, 6>(vec2(0.0, -1.0), vec2(0.0, 1.0), vec2(1.0, -1.0), vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, -1.0));
  let k = cs[corner];
  let c = select(c0, c1, k.x > 0.5);
  let off = nrm * k.y * F.viewport.zw * 2.0 * c.w;
  o.pos = vec4<f32>(c.xy + off, c.z, c.w);
  o.color = color;
  o.side = k.y;
  return o;
}
fn degenerate() -> RO {
  var o: RO;
  o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  o.color = vec4<f32>(0.0);
  o.side = 0.0;
  return o;
}
@fragment fn fs(i: RO) -> @location(0) vec4<f32> {
  let a = 1.0 - smoothstep(0.55, 1.0, abs(i.side));
  let c = i.color * a;
  return vec4<f32>(c.rgb * c.a, c.a);
}
`;

export const TRAILS_WGSL = /* wgsl */ `
${FRAME_WGSL}
${RIBBON}
struct TU {
  count: u32, trail: u32, head: u32, colorMode: u32,
  width: f32, alpha: f32, p0: f32, p1: f32,
};
@group(1) @binding(0) var<uniform> T: TU;
@group(1) @binding(1) var<storage, read> state: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read> hist: array<vec4<f32>>;

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> RO {
  let seg = vi / 6u;
  let corner = vi % 6u;
  let age = state[ii].w;
  if (seg + 1u >= T.trail || f32(seg + 1u) > age) { return degenerate(); }
  let L = T.trail;
  let h0 = (T.head + L - seg) % L;
  let h1 = (T.head + L - seg - 1u) % L;
  let p0 = hist[h0 * T.count + ii].xyz;
  let p1 = hist[h1 * T.count + ii].xyz;
  if (distance(p0, p1) > 12.0) { return degenerate(); }
  let fade = 1.0 - f32(seg) / f32(L - 1u);
  let U = max(F.flow.x, 1e-5);
  var rgb = vec3<f32>(0.92, 0.95, 1.0);
  if (T.colorMode == 1u) { rgb = cm_turbo(length(sVel(p0).xyz) / U * 0.7); }
  return ribbon(p0, p1, corner, T.width, vec4<f32>(rgb, T.alpha * fade * fade));
}
`;

export const STREAM_WGSL = /* wgsl */ `
${FRAME_WGSL}
${RIBBON}
struct SU {
  seeds: u32, points: u32, p0: u32, p1: u32,
  a: vec4<f32>, b: vec4<f32>,
  h: f32, width: f32, alpha: f32, p3: f32,
};
@group(1) @binding(0) var<uniform> S: SU;
@group(1) @binding(1) var<storage, read> pts: array<vec4<f32>>;

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> RO {
  let seg = vi / 6u;
  let corner = vi % 6u;
  if (seg + 1u >= S.points) { return degenerate(); }
  let a = pts[ii * S.points + seg];
  let b = pts[ii * S.points + seg + 1u];
  if (a.w < 0.0 || b.w < 0.0) { return degenerate(); }
  let U = max(F.flow.x, 1e-5);
  let rgb = cm_turbo(a.w / U * 0.7);
  return ribbon(a.xyz, b.xyz, corner, S.width, vec4<f32>(rgb, S.alpha));
}
`;

export const SEGMENTS_WGSL = /* wgsl */ `
${FRAME_WGSL}
${RIBBON}
struct Seg { a: vec4<f32>, b: vec4<f32>, color: vec4<f32> };
@group(1) @binding(0) var<storage, read> segs: array<Seg>;
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> RO {
  let s = segs[ii];
  return ribbon(s.a.xyz, s.b.xyz, vi, s.a.w * F.extra.y, s.color);
}
`;

export const GROUND_WGSL = /* wgsl */ `
${FRAME_WGSL}
struct VO { @builtin(position) pos: vec4<f32>, @location(0) wp: vec3<f32> };
struct GU { x0: f32, x1: f32, z0: f32, z1: f32, moving: f32, spacing: f32, p0: f32, p1: f32 };
@group(1) @binding(0) var<uniform> G: GU;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VO {
  var q = array<vec2<f32>, 6>(vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0));
  let t = q[vi];
  var o: VO;
  o.wp = vec3<f32>(mix(G.x0, G.x1, t.x), -0.02, mix(G.z0, G.z1, t.y));
  o.pos = F.viewProj * vec4<f32>(o.wp, 1.0);
  return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4<f32> {
  let p = i.wp;
  let d = F.dims.xyz;
  let inside = p.x >= 0.0 && p.x <= d.x && p.z >= select(0.0, -1e9, F.dims.w > 0.5) && p.z <= select(d.z, 1e9, F.dims.w > 0.5);
  let g = p.xz / G.spacing;
  let fw = fwidth(g);
  let gl = abs(fract(g - 0.5) - 0.5) / max(fw, vec2<f32>(1e-4));
  let line = 1.0 - min(min(gl.x, gl.y), 1.0);
  var col = vec3<f32>(0.075, 0.082, 0.095);
  if (inside) { col = vec3<f32>(0.1, 0.11, 0.125); }
  col += vec3<f32>(0.06, 0.07, 0.085) * line;
  if (inside && G.moving > 0.5) {
    let s = fract((p.x - F.extra.x) / (G.spacing * 2.0));
    let belt = smoothstep(0.0, 0.04, s) * (1.0 - smoothstep(0.08, 0.12, s));
    col += vec3<f32>(0.05, 0.09, 0.12) * belt;
  }
  let cx = clamp(p.x, 0.0, d.x);
  let dist = length(vec2<f32>(p.x - cx, 0.0)) / d.x;
  let fade = 1.0 - smoothstep(0.0, 0.5, dist);
  return vec4<f32>(col * fade + vec3<f32>(0.043, 0.047, 0.055) * (1.0 - fade), 1.0);
}
`;
