/**
 * Perceptual colormaps. The same polynomial fits are used in WGSL, GLSL and JS (for legends), so
 * the legend always matches the rendered pixels exactly.
 *  0 turbo (Google, Mikhailov 2019 polynomial fit)
 *  1 viridis (polynomial fit, Matt Zucker)
 *  2 coolwarm (Moreland diverging, piecewise-linear through its control points)
 *  3 inferno (polynomial fit)
 */
export const COLORMAPS = ['turbo', 'viridis', 'coolwarm', 'inferno'] as const;
export type ColormapName = (typeof COLORMAPS)[number];

const TURBO_R = [0.13572138, 4.6153926, -42.66032258, 132.13108234, -152.94239396, 59.28637943];
const TURBO_G = [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604];
const TURBO_B = [0.1066733, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973];
const VIR = {
  c0: [0.2777273272234177, 0.005407344544966578, 0.3340998053353061],
  c1: [0.1050930431085774, 1.404613529898575, 1.384590162594685],
  c2: [-0.3308618287255563, 0.214847559468213, 0.09509516302823659],
  c3: [-4.634230498983486, -5.799100973351585, -19.33244095627987],
  c4: [6.228269936347081, 14.17993336680509, 56.69055260068105],
  c5: [4.776384997670288, -13.74514537774601, -65.35303263337234],
  c6: [-5.435455855934631, 4.645852612178535, 26.3124352495832],
};
const INF = {
  c0: [0.0002189403691192265, 0.001651004631001012, -0.01948089843709184],
  c1: [0.1065134194856116, 0.5639564367884091, 3.932712388889277],
  c2: [11.60249308247187, -3.972853965665698, -15.9423941062914],
  c3: [-41.70399613139459, 17.43639888205313, 44.35414519872813],
  c4: [77.162935699427, -33.40235894210092, -81.80730925738993],
  c5: [-71.31942824499214, 32.62606426397723, 73.20951985803202],
  c6: [25.13112622477341, -12.24266895238567, -23.07032500287172],
};
const COOLWARM: [number, number, number][] = [
  [0.230, 0.299, 0.754], [0.406, 0.537, 0.934], [0.602, 0.731, 0.999], [0.788, 0.846, 0.939],
  [0.867, 0.865, 0.865], [0.958, 0.784, 0.694], [0.968, 0.608, 0.476], [0.866, 0.389, 0.304], [0.706, 0.016, 0.150],
];

const f = (v: number) => v.toPrecision(10);
const vec3 = (a: number[]) => `vec3<f32>(${a.map(f).join(', ')})`;
const vec3gl = (a: number[]) => `vec3(${a.map(f).join(', ')})`;

function poly6(c: typeof VIR, v3: (a: number[]) => string, t: string): string {
  return `(${v3(c.c0)} + ${t} * (${v3(c.c1)} + ${t} * (${v3(c.c2)} + ${t} * (${v3(c.c3)} + ${t} * (${v3(c.c4)} + ${t} * (${v3(c.c5)} + ${t} * ${v3(c.c6)}))))))`;
}

export const COLORMAP_WGSL = /* wgsl */ `
fn cm_turbo(x: f32) -> vec3<f32> {
  let t = clamp(x, 0.0, 1.0);
  let v4 = vec4<f32>(1.0, t, t * t, t * t * t);
  let v2 = v4.zw * v4.z;
  return vec3<f32>(
    dot(v4, vec4<f32>(${TURBO_R.slice(0, 4).map(f).join(', ')})) + dot(v2, vec2<f32>(${TURBO_R.slice(4).map(f).join(', ')})),
    dot(v4, vec4<f32>(${TURBO_G.slice(0, 4).map(f).join(', ')})) + dot(v2, vec2<f32>(${TURBO_G.slice(4).map(f).join(', ')})),
    dot(v4, vec4<f32>(${TURBO_B.slice(0, 4).map(f).join(', ')})) + dot(v2, vec2<f32>(${TURBO_B.slice(4).map(f).join(', ')})));
}
fn cm_viridis(x: f32) -> vec3<f32> {
  let t = clamp(x, 0.0, 1.0);
  return clamp(${poly6(VIR, vec3, 't')}, vec3<f32>(0.0), vec3<f32>(1.0));
}
fn cm_inferno(x: f32) -> vec3<f32> {
  let t = clamp(x, 0.0, 1.0);
  return clamp(${poly6(INF, vec3, 't')}, vec3<f32>(0.0), vec3<f32>(1.0));
}
fn cm_coolwarm(x: f32) -> vec3<f32> {
  let t = clamp(x, 0.0, 1.0) * 8.0;
  let i = min(u32(floor(t)), 7u);
  let fr = t - f32(i);
  var c = array<vec3<f32>, 9>(${COOLWARM.map((c) => vec3(c)).join(', ')});
  return mix(c[i], c[i + 1u], fr);
}
fn colormap(id: u32, x: f32) -> vec3<f32> {
  switch (id) {
    case 1u: { return cm_viridis(x); }
    case 2u: { return cm_coolwarm(x); }
    case 3u: { return cm_inferno(x); }
    default: { return cm_turbo(x); }
  }
}
`;

export const COLORMAP_GLSL = /* glsl */ `
vec3 cm_turbo(float x) {
  float t = clamp(x, 0.0, 1.0);
  vec4 v4 = vec4(1.0, t, t * t, t * t * t);
  vec2 v2 = v4.zw * v4.z;
  return vec3(
    dot(v4, vec4(${TURBO_R.slice(0, 4).map(f).join(', ')})) + dot(v2, vec2(${TURBO_R.slice(4).map(f).join(', ')})),
    dot(v4, vec4(${TURBO_G.slice(0, 4).map(f).join(', ')})) + dot(v2, vec2(${TURBO_G.slice(4).map(f).join(', ')})),
    dot(v4, vec4(${TURBO_B.slice(0, 4).map(f).join(', ')})) + dot(v2, vec2(${TURBO_B.slice(4).map(f).join(', ')})));
}
vec3 cm_viridis(float x) { float t = clamp(x, 0.0, 1.0); return clamp(${poly6(VIR, vec3gl, 't')}, 0.0, 1.0); }
vec3 cm_inferno(float x) { float t = clamp(x, 0.0, 1.0); return clamp(${poly6(INF, vec3gl, 't')}, 0.0, 1.0); }
vec3 cm_coolwarm(float x) {
  float t = clamp(x, 0.0, 1.0) * 8.0;
  int i = int(min(floor(t), 7.0));
  float fr = t - float(i);
  vec3 c[9] = vec3[9](${COOLWARM.map((c) => vec3gl(c)).join(', ')});
  return mix(c[i], c[i + 1], fr);
}
vec3 colormap(int id, float x) {
  if (id == 1) return cm_viridis(x);
  if (id == 2) return cm_coolwarm(x);
  if (id == 3) return cm_inferno(x);
  return cm_turbo(x);
}
`;

function polyJS(c: typeof VIR, t: number): [number, number, number] {
  const out: [number, number, number] = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const v = c.c0[k] + t * (c.c1[k] + t * (c.c2[k] + t * (c.c3[k] + t * (c.c4[k] + t * (c.c5[k] + t * c.c6[k])))));
    out[k] = Math.min(1, Math.max(0, v));
  }
  return out;
}

export function colormapJS(name: ColormapName, x: number): [number, number, number] {
  const t = Math.min(1, Math.max(0, x));
  switch (name) {
    case 'viridis': return polyJS(VIR, t);
    case 'inferno': return polyJS(INF, t);
    case 'coolwarm': {
      const s = t * 8;
      const i = Math.min(Math.floor(s), 7);
      const fr = s - i;
      const a = COOLWARM[i], b = COOLWARM[i + 1];
      return [a[0] + (b[0] - a[0]) * fr, a[1] + (b[1] - a[1]) * fr, a[2] + (b[2] - a[2]) * fr];
    }
    default: {
      const p = (c: number[]) => c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * (c[4] + t * c[5]))));
      return [p(TURBO_R), p(TURBO_G), p(TURBO_B)].map((v) => Math.min(1, Math.max(0, v))) as [number, number, number];
    }
  }
}

export function colormapCSS(name: ColormapName, stops = 16): string {
  const parts: string[] = [];
  for (let i = 0; i <= stops; i++) {
    const [r, g, b] = colormapJS(name, i / stops);
    parts.push(`rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)}) ${((i / stops) * 100).toFixed(1)}%`);
  }
  return `linear-gradient(90deg, ${parts.join(', ')})`;
}
