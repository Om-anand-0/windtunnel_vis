import * as THREE from 'three';
import { FullscreenPass, GLTex, makeTex, SolverGL } from '../solver/SolverGL';
import type { MeshData } from '../voxelize/mesh';
import { COLORMAP_GLSL } from './colormaps';
import type { RenderState } from './renderState';

/**
 * WebGL2 fallback renderer (2D mode). Three.js draws the scene; the solver and the tracer GPGPU
 * passes run as raw GL on the same context and are exposed to Three.js as ExternalTextures.
 */

const HASH = /* glsl */ `
uint pcg(uint v) { uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
uint rs;
float rnd() { rs = pcg(rs); return float(rs) / 4294967295.0; }
`;

const ADVECT = /* glsl */ `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D STATE; uniform sampler2D VEL; uniform sampler2D FLAGS;
uniform ivec2 N; uniform int W; uniform int count; uniform int emitter; uniform int nozzles; uniform int fill; uniform int advance;
uniform float steps; uniform float maxAge; uniform float trail; uniform uint seed;
uniform vec2 rakeA; uniform vec2 rakeB;
out vec4 o;
${HASH}
vec2 vel(vec2 p) { return texture(VEL, p / vec2(N)).xy; }
bool solidAt(vec2 p) { ivec2 c = ivec2(floor(p)); if (c.x < 0 || c.y < 0 || c.x >= N.x || c.y >= N.y) return false; return texelFetch(FLAGS, c, 0).r > 0.5; }
vec2 spawn() {
  if (emitter == 0) return vec2(1.0 + rnd() * 2.0, 0.5 + rnd() * (float(N.y) - 1.0));
  int k = min(int(rnd() * float(nozzles)), nozzles - 1);
  float t = (float(k) + 0.5) / float(nozzles);
  return mix(rakeA, rakeB, t) + vec2(rnd() - 0.5, rnd() - 0.5) * 0.35;
}
void main() {
  ivec2 tc = ivec2(gl_FragCoord.xy);
  int id = tc.x + tc.y * W;
  vec4 s = texelFetch(STATE, tc, 0);
  if (id >= count) { o = vec4(-10.0, -10.0, 0.0, 0.0); return; }
  rs = pcg(uint(id) ^ pcg(seed));
  if (fill == 1) {
    if (emitter == 0) {
      vec2 p = vec2(1.0 + rnd() * (float(N.x) - 2.0), 0.5 + rnd() * (float(N.y) - 1.0));
      o = vec4(p, trail, 0.0);
    } else {
      o = vec4(spawn(), -floor(rnd() * maxAge), 0.0);
    }
    return;
  }
  if (advance == 0) { o = s; return; }
  if (s.z < 0.0) {
    s.z += 1.0;
    if (s.z >= 0.0) s = vec4(spawn(), 0.0, 0.0);
  } else {
    vec2 p = s.xy;
    float disp = length(vel(p)) * steps;
    int nsub = int(clamp(ceil(disp / 0.6), 1.0, 24.0));
    float h = steps / float(nsub);
    for (int k = 0; k < 24; k++) {
      if (k >= nsub) break;
      vec2 a = vel(p);
      vec2 b = vel(p + a * 0.5 * h);
      p += b * h;
    }
    s = vec4(p, s.z + 1.0, 0.0);
    if (p.x < 0.0 || p.y < 0.0 || p.x >= float(N.x) - 1.0 || p.y >= float(N.y) || solidAt(p) || s.z > maxAge * 4.0) s = vec4(spawn(), 0.0, 0.0);
  }
  o = s;
}`;

const PROBES = /* glsl */ `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D VEL; uniform sampler2D MEAN; uniform vec2 N; uniform vec2 P[8];
out vec4 o;
void main() {
  ivec2 t = ivec2(gl_FragCoord.xy);
  vec2 uv = P[t.x] / N;
  o = t.y == 0 ? texture(VEL, uv) : texture(MEAN, uv);
}`;

const COPY = /* glsl */ `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D SRC; uniform int yOff;
out vec4 o;
void main() { o = texelFetch(SRC, ivec2(gl_FragCoord.x, gl_FragCoord.y - float(yOff)), 0); }`;

const STREAMLINES = /* glsl */ `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D VEL; uniform sampler2D MEAN; uniform sampler2D FLAGS; uniform ivec2 N; uniform int seeds; uniform int useMean;
uniform vec2 rakeA; uniform vec2 rakeB; uniform float h; uniform float U;
out vec4 o;
vec2 vel(vec2 p) { return useMean == 1 ? texture(MEAN, p / vec2(N)).xy : texture(VEL, p / vec2(N)).xy; }
bool bad(vec2 p) {
  if (p.x < 0.0 || p.y < 0.0 || p.x >= float(N.x) || p.y >= float(N.y)) return true;
  return texelFetch(FLAGS, ivec2(floor(p)), 0).r > 0.5;
}
void main() {
  int k = int(gl_FragCoord.x);
  int s = int(gl_FragCoord.y);
  vec2 p = mix(rakeA, rakeB, (float(s) + 0.5) / float(seeds));
  for (int j = 0; j < 2048; j++) {
    if (j >= k) break;
    vec2 v = vel(p);
    float sp = length(v);
    if (bad(p) || sp < U * 1e-3) { o = vec4(p, -1.0, 0.0); return; }
    vec2 vm = vel(p + v / sp * 0.5 * h);
    float lm = length(vm);
    if (lm < 1e-9) { o = vec4(p, -1.0, 0.0); return; }
    p += vm / lm * h;
  }
  float sp = length(vel(p));
  o = vec4(p, bad(p) ? -1.0 : sp, 0.0);
}`;

const RIBBON_VS = /* glsl */ `
uniform mat4 viewProj;
uniform vec2 viewport;
out vec4 vColor;
out float vSide;
void ribbon(vec3 p0, vec3 p1, int corner, float width, vec4 color) {
  vec4 c0 = viewProj * vec4(p0, 1.0);
  vec4 c1 = viewProj * vec4(p1, 1.0);
  vec2 s0 = c0.xy / c0.w * viewport * 0.5;
  vec2 s1 = c1.xy / c1.w * viewport * 0.5;
  vec2 d = s1 - s0;
  float l = length(d);
  d = l > 1e-5 ? d / l : vec2(1.0, 0.0);
  vec2 nrm = vec2(-d.y, d.x) * width * 0.5;
  vec2 cs[6] = vec2[6](vec2(0.0, -1.0), vec2(0.0, 1.0), vec2(1.0, -1.0), vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, -1.0));
  vec2 k = cs[corner];
  vec4 c = k.x > 0.5 ? c1 : c0;
  vec2 off = nrm * k.y / viewport * 2.0 * c.w;
  gl_Position = vec4(c.xy + off, c.z, c.w);
  vColor = color;
  vSide = k.y;
}
void degenerate() { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vColor = vec4(0.0); vSide = 0.0; }
`;

const RIBBON_FS = /* glsl */ `
precision highp float;
in vec4 vColor;
in float vSide;
out vec4 outColor;
void main() {
  float a = 1.0 - smoothstep(0.55, 1.0, abs(vSide));
  vec4 c = vColor * a;
  outColor = vec4(c.rgb * c.a, c.a);
}`;

const TRAILS_VS = /* glsl */ `
precision highp float; precision highp int; precision highp sampler2D;
in vec3 position;
uniform sampler2D HIST; uniform sampler2D STATE; uniform sampler2D VEL;
uniform int W; uniform int H; uniform int T; uniform int head; uniform int colorMode;
uniform float width; uniform float alpha; uniform vec2 N; uniform float U;
${COLORMAP_GLSL}
${RIBBON_VS}
void main() {
  int ii = gl_InstanceID;
  int seg = gl_VertexID / 6;
  int corner = gl_VertexID % 6;
  ivec2 st = ivec2(ii % W, ii / W);
  float age = texelFetch(STATE, st, 0).z;
  if (seg + 1 >= T || float(seg + 1) > age) { degenerate(); return; }
  int h0 = (head + T - seg) % T;
  int h1 = (head + T - seg - 1) % T;
  vec2 p0 = texelFetch(HIST, ivec2(st.x, st.y + h0 * H), 0).xy;
  vec2 p1 = texelFetch(HIST, ivec2(st.x, st.y + h1 * H), 0).xy;
  if (distance(p0, p1) > 12.0) { degenerate(); return; }
  float fade = 1.0 - float(seg) / float(T - 1);
  vec3 rgb = vec3(0.92, 0.95, 1.0);
  if (colorMode == 1) rgb = cm_turbo(length(texture(VEL, p0 / N).xy) / U * 0.7);
  ribbon(vec3(p0, 0.5), vec3(p1, 0.5), corner, width, vec4(rgb, alpha * fade * fade));
}`;

const STREAM_VS = /* glsl */ `
precision highp float; precision highp int; precision highp sampler2D;
in vec3 position;
uniform sampler2D PTS; uniform int P; uniform float width; uniform float alpha; uniform float U;
${COLORMAP_GLSL}
${RIBBON_VS}
void main() {
  int s = gl_InstanceID;
  int seg = gl_VertexID / 6;
  int corner = gl_VertexID % 6;
  if (seg + 1 >= P) { degenerate(); return; }
  vec4 a = texelFetch(PTS, ivec2(seg, s), 0);
  vec4 b = texelFetch(PTS, ivec2(seg + 1, s), 0);
  if (a.z < 0.0 || b.z < 0.0) { degenerate(); return; }
  ribbon(vec3(a.xy, 0.5), vec3(b.xy, 0.5), corner, width, vec4(cm_turbo(a.z / U * 0.7), alpha));
}`;

const SLICE_VS = /* glsl */ `
precision highp float;
in vec3 position;
uniform mat4 viewProj;
out vec2 gp;
void main() { gp = position.xy; gl_Position = viewProj * vec4(position, 1.0); }`;

const SLICE_FS = /* glsl */ `
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D VEL; uniform sampler2D VORT; uniform sampler2D MEAN;
uniform vec2 N; uniform float U; uniform float Lref; uniform float rhoRef;
uniform int field; uniform int cmap; uniform int flags; uniform float vmin; uniform float vmax; uniform float opacity; uniform float qThr;
in vec2 gp;
out vec4 outColor;
${COLORMAP_GLSL}
void main() {
  vec2 uv = gp / N;
  vec4 vel = texture(VEL, uv);
  vec4 vo = texture(VORT, uv);
  vec4 mn = texture(MEAN, uv);
  bool useMean = (flags & 1) != 0;
  float Uu = max(U, 1e-5);
  float v;
  if (field == 0) v = (useMean ? length(mn.xy) : length(vel.xy)) / Uu;
  else if (field == 1) v = (useMean ? mn.x : vel.x) / Uu;
  else if (field == 2) v = 2.0 * ((useMean ? mn.z : vel.z) - rhoRef) / (3.0 * Uu * Uu);
  else if (field == 3) v = vo.x * Lref / Uu;
  else if (field == 4) v = vo.z;
  else v = vo.y * Lref * Lref / (Uu * Uu);
  float rc = mn.x / Uu;
  float rcw = fwidth(rc);
  float qn = vo.y * Lref * Lref / (Uu * Uu) - qThr;
  float qw = fwidth(qn);
  float sw = fwidth(vel.w);
  vec3 col = colormap(cmap, (v - vmin) / (vmax - vmin));
  if ((flags & 2) != 0) {
    if (rc < 0.0) col = mix(col, vec3(1.0), 0.18);
    col = mix(vec3(1.0), col, smoothstep(0.5, 1.5, abs(rc) / max(rcw, 1e-6)));
  }
  if ((flags & 4) != 0) col = mix(vec3(0.05), col, smoothstep(0.5, 1.5, abs(qn) / max(qw, 1e-6)));
  float edge = smoothstep(0.35, 0.65, vel.w);
  float outline = 1.0 - smoothstep(0.0, 1.5, abs(vel.w - 0.5) / max(sw, 1e-4));
  col = mix(col, vec3(0.11, 0.12, 0.14), edge);
  col = mix(col, vec3(0.85, 0.88, 0.92), outline * 0.8);
  outColor = vec4(col * opacity, opacity);
}`;

const MESH_VS = /* glsl */ `
precision highp float;
in vec3 position; in vec3 normal;
uniform mat4 viewProj; uniform mat4 model; uniform mat4 normalM;
out vec3 world; out vec3 vn;
void main() {
  vec4 w = model * vec4(position, 1.0);
  world = w.xyz;
  vn = normalize((normalM * vec4(normal, 0.0)).xyz);
  gl_Position = viewProj * w;
}`;

const MESH_FS = /* glsl */ `
precision highp float; precision highp int; precision highp sampler2D;
uniform sampler2D VEL; uniform sampler2D MEAN; uniform vec2 N; uniform float U; uniform float rhoRef; uniform vec3 eye;
uniform int mode; uniform int useMean; uniform float cpMin; uniform float cpMax; uniform float opacity;
in vec3 world; in vec3 vn;
out vec4 outColor;
${COLORMAP_GLSL}
float solidAt(vec2 p) { return texture(VEL, p / N).w; }
void main() {
  vec3 n = normalize(vn);
  if (!gl_FrontFacing) n = -n;
  vec3 v = normalize(eye - world);
  vec3 key = normalize(vec3(-0.45, 0.8, 0.55));
  float lam = max(dot(n, key), 0.0);
  float hemi = 0.5 + 0.5 * n.y;
  float fres = pow(1.0 - max(dot(n, v), 0.0), 4.0);
  vec3 col;
  if (mode == 1) {
    vec2 n2 = length(n.xy) > 1e-3 ? normalize(n.xy) : vec2(0.0, 1.0);
    vec2 q = world.xy + n2 * 1.5;
    if (solidAt(q) > 0.5) q = world.xy - n2 * 1.5;
    vec4 s = useMean == 1 ? texture(MEAN, q / N) : texture(VEL, q / N);
    float cp = 2.0 * (s.z - rhoRef) / (3.0 * max(U * U, 1e-10));
    col = colormap(0, (cp - cpMin) / (cpMax - cpMin)) * (0.62 + 0.3 * lam + 0.12 * hemi) + vec3(0.15) * fres;
  } else {
    vec3 base = vec3(0.46, 0.5, 0.56);
    vec3 hh = normalize(key + v);
    float spec = pow(max(dot(n, hh), 0.0), 60.0) * 0.6;
    col = base * (0.2 + 0.65 * lam + 0.22 * hemi) + vec3(spec) + vec3(0.35, 0.45, 0.6) * fres * 0.6;
  }
  outColor = vec4(col * opacity, opacity);
}`;

const GROUND_VS = /* glsl */ `
precision highp float;
in vec3 position;
uniform mat4 viewProj;
out vec3 wp;
void main() { wp = position; gl_Position = viewProj * vec4(position, 1.0); }`;

const GROUND_FS = /* glsl */ `
precision highp float;
uniform vec2 N; uniform float spacing; uniform float moving; uniform float offset;
in vec3 wp;
out vec4 outColor;
void main() {
  bool inside = wp.x >= 0.0 && wp.x <= N.x;
  vec2 g = wp.xz / spacing;
  vec2 fw = fwidth(g);
  vec2 gl2 = abs(fract(g - 0.5) - 0.5) / max(fw, vec2(1e-4));
  float line = 1.0 - min(min(gl2.x, gl2.y), 1.0);
  vec3 col = inside ? vec3(0.1, 0.11, 0.125) : vec3(0.075, 0.082, 0.095);
  col += vec3(0.06, 0.07, 0.085) * line;
  if (inside && moving > 0.5) {
    float s = fract((wp.x - offset) / (spacing * 2.0));
    col += vec3(0.05, 0.09, 0.12) * smoothstep(0.0, 0.04, s) * (1.0 - smoothstep(0.08, 0.12, s));
  }
  float dist = abs(wp.x - clamp(wp.x, 0.0, N.x)) / N.x;
  float fade = 1.0 - smoothstep(0.0, 0.5, dist);
  outColor = vec4(col * fade + vec3(0.043, 0.047, 0.055) * (1.0 - fade), 1.0);
}`;

const PREMUL = {
  blending: THREE.CustomBlending,
  blendSrc: THREE.OneFactor,
  blendDst: THREE.OneMinusSrcAlphaFactor,
  blendSrcAlpha: THREE.OneFactor,
  blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
  transparent: true,
  depthWrite: false,
} as const;

export class RendererGL {
  readonly three: THREE.WebGLRenderer;
  readonly gl: WebGL2RenderingContext;
  private scene = new THREE.Scene();
  private cam = new THREE.Camera();
  private solver: SolverGL | null = null;
  private ext = new Map<WebGLTexture, THREE.ExternalTexture>();

  private slice: THREE.Mesh<THREE.PlaneGeometry, THREE.RawShaderMaterial>;
  private ground: THREE.Mesh<THREE.PlaneGeometry, THREE.RawShaderMaterial>;
  private box: THREE.LineSegments;
  private rakeLine: THREE.Line;
  private rakeHandle: THREE.Mesh;
  private body: THREE.Mesh<THREE.BufferGeometry, THREE.RawShaderMaterial> | null = null;
  private meshMat: THREE.RawShaderMaterial;

  // tracers
  private advect: FullscreenPass;
  private copy: FullscreenPass;
  private stream: FullscreenPass;
  private probePass: FullscreenPass;
  private probeTex: { tex: GLTex; fbo: WebGLFramebuffer } | null = null;
  private probeFrame = 0;
  private probeMarks: THREE.LineSegments;
  /** latest probe samples: per probe [ux, uy, uz, ρ] instantaneous and mean */
  probeData: { inst: number[]; mean: number[] }[] = [];
  private vao: WebGLVertexArrayObject;
  private pState: [GLTex, GLTex] | null = null;
  private pFbo: [WebGLFramebuffer, WebGLFramebuffer] | null = null;
  private pHist: { tex: GLTex; fbo: WebGLFramebuffer } | null = null;
  private pCur = 0;
  private pW = 512;
  private pH = 1;
  private pCount = 0;
  private pTrail = 0;
  private head = 0;
  private needFill = true;
  private seed = 1;
  private trails: THREE.Mesh<THREE.InstancedBufferGeometry, THREE.RawShaderMaterial> | null = null;
  private slTex: { tex: GLTex; fbo: WebGLFramebuffer; P: number; S: number } | null = null;
  private slMesh: THREE.Mesh<THREE.InstancedBufferGeometry, THREE.RawShaderMaterial> | null = null;

  constructor(readonly canvas: HTMLCanvasElement) {
    this.three = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.three.setClearColor(0x0b0c0e, 1);
    this.three.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.gl = this.three.getContext() as WebGL2RenderingContext;
    this.cam.matrixAutoUpdate = false;
    this.cam.matrixWorldAutoUpdate = false;

    const mat = (vs: string, fs: string, uniforms: Record<string, THREE.IUniform>, extra: Partial<THREE.ShaderMaterialParameters> = {}) =>
      new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: vs, fragmentShader: fs, uniforms, side: THREE.DoubleSide, ...extra });

    const common = () => ({ viewProj: { value: new THREE.Matrix4() } });
    this.slice = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat(SLICE_VS, SLICE_FS, {
      ...common(), VEL: { value: null }, VORT: { value: null }, MEAN: { value: null }, N: { value: new THREE.Vector2() }, U: { value: 0.05 }, Lref: { value: 100 },
      rhoRef: { value: 1 }, field: { value: 0 }, cmap: { value: 0 }, flags: { value: 0 }, vmin: { value: 0 }, vmax: { value: 1.5 }, opacity: { value: 1 }, qThr: { value: 1 },
    }));
    this.slice.renderOrder = 2;
    this.ground = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat(GROUND_VS, GROUND_FS, {
      ...common(), N: { value: new THREE.Vector2() }, spacing: { value: 32 }, moving: { value: 1 }, offset: { value: 0 },
    }));
    this.ground.renderOrder = 0;
    this.meshMat = mat(MESH_VS, MESH_FS, {
      ...common(), model: { value: new THREE.Matrix4() }, normalM: { value: new THREE.Matrix4() }, VEL: { value: null }, MEAN: { value: null },
      N: { value: new THREE.Vector2() }, U: { value: 0.05 }, rhoRef: { value: 1 }, eye: { value: new THREE.Vector3() }, mode: { value: 1 }, useMean: { value: 1 },
      cpMin: { value: -1.5 }, cpMax: { value: 1 }, opacity: { value: 1 },
    });
    this.box = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x5f7593, transparent: true, opacity: 0.7 }));
    this.box.renderOrder = 5;
    this.rakeLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), new THREE.LineBasicMaterial({ color: 0xffc233, depthTest: false }));
    this.rakeLine.renderOrder = 10;
    this.rakeHandle = new THREE.Mesh(new THREE.CircleGeometry(1, 24), new THREE.MeshBasicMaterial({ color: 0xffd166, depthTest: false }));
    this.rakeHandle.renderOrder = 11;
    for (const o of [this.slice, this.ground, this.box, this.rakeLine, this.rakeHandle]) {
      o.frustumCulled = false;
      o.matrixAutoUpdate = false;
      this.scene.add(o);
    }
    this.advect = new FullscreenPass(this.gl, ADVECT);
    this.copy = new FullscreenPass(this.gl, COPY);
    this.stream = new FullscreenPass(this.gl, STREAMLINES);
    this.probePass = new FullscreenPass(this.gl, PROBES);
    this.probeMarks = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false }));
    this.probeMarks.renderOrder = 12;
    this.probeMarks.frustumCulled = false;
    this.scene.add(this.probeMarks);
    this.vao = this.gl.createVertexArray()!;
  }

  private wrap(t: GLTex): THREE.ExternalTexture {
    let e = this.ext.get(t.tex);
    if (!e) {
      e = new THREE.ExternalTexture(t.tex);
      this.ext.set(t.tex, e);
    }
    return e;
  }

  setSolver(s: SolverGL) {
    this.solver = s;
    const { nx, ny } = s.dims;
    this.slice.geometry.dispose();
    const pg = new THREE.PlaneGeometry(nx, ny);
    pg.translate(nx / 2, ny / 2, 0.5);
    this.slice.geometry = pg;
    const gg = new THREE.PlaneGeometry(nx * 2.2, Math.max(nx * 0.5, 120));
    gg.rotateX(-Math.PI / 2);
    gg.translate(nx * 0.5, -0.02, 0);
    this.ground.geometry.dispose();
    this.ground.geometry = gg;
    const pts: number[] = [];
    const E = (a: number[], b: number[]) => pts.push(...a, ...b);
    E([0, 0, 0.5], [nx, 0, 0.5]); E([0, ny, 0.5], [nx, ny, 0.5]); E([0, 0, 0.5], [0, ny, 0.5]); E([nx, 0, 0.5], [nx, ny, 0.5]);
    this.box.geometry.dispose();
    this.box.geometry = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    this.needFill = true;
  }

  setMesh(mesh: MeshData | null) {
    if (this.body) {
      this.scene.remove(this.body);
      this.body.geometry.dispose();
      this.body = null;
    }
    if (!mesh) return;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(mesh.normals, 3));
    g.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
    this.body = new THREE.Mesh(g, this.meshMat);
    this.body.frustumCulled = false;
    this.body.renderOrder = 1;
    this.scene.add(this.body);
  }

  refillParticles() {
    this.needFill = true;
  }

  pixelSize(): [number, number] {
    const v = new THREE.Vector2();
    this.three.getDrawingBufferSize(v);
    return [v.x, v.y];
  }

  private ensureTracers(count: number, trail: number) {
    if (this.pState && this.pCount === count && this.pTrail === trail) return;
    const gl = this.gl;
    if (this.pState) {
      for (const t of this.pState) gl.deleteTexture(t.tex);
      for (const f of this.pFbo!) gl.deleteFramebuffer(f);
      gl.deleteTexture(this.pHist!.tex.tex);
      gl.deleteFramebuffer(this.pHist!.fbo);
      this.ext.clear();
    }
    this.pW = 512;
    this.pH = Math.ceil(count / this.pW);
    const mk = () => makeTex(gl, this.pW, this.pH, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    this.pState = [mk(), mk()];
    const fbo = (t: GLTex) => {
      const f = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, f);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t.tex, 0);
      return f;
    };
    this.pFbo = [fbo(this.pState[0]), fbo(this.pState[1])];
    const ht = makeTex(gl, this.pW, this.pH * trail, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    this.pHist = { tex: ht, fbo: fbo(ht) };
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.pCount = count;
    this.pTrail = trail;
    this.head = 0;
    this.needFill = true;
    if (this.trails) {
      this.scene.remove(this.trails);
      this.trails.geometry.dispose();
    }
    const geo = new THREE.InstancedBufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array((trail - 1) * 6 * 3), 3));
    geo.instanceCount = count;
    const m = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader: TRAILS_VS, fragmentShader: RIBBON_FS, ...PREMUL, depthTest: false,
      uniforms: {
        viewProj: { value: new THREE.Matrix4() }, viewport: { value: new THREE.Vector2() }, HIST: { value: null }, STATE: { value: null }, VEL: { value: null },
        W: { value: this.pW }, H: { value: this.pH }, T: { value: trail }, head: { value: 0 }, colorMode: { value: 0 }, width: { value: 1.2 }, alpha: { value: 0.4 },
        N: { value: new THREE.Vector2() }, U: { value: 0.05 },
      },
    });
    this.trails = new THREE.Mesh(geo, m);
    this.trails.frustumCulled = false;
    this.trails.renderOrder = 8;
    this.scene.add(this.trails);
  }

  private ensureStreamlines(P: number, S: number) {
    if (this.slTex && this.slTex.P === P && this.slTex.S === S) return;
    const gl = this.gl;
    if (this.slTex) {
      gl.deleteTexture(this.slTex.tex.tex);
      gl.deleteFramebuffer(this.slTex.fbo);
    }
    const t = makeTex(gl, P, S, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    const f = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t.tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.slTex = { tex: t, fbo: f, P, S };
    if (this.slMesh) {
      this.scene.remove(this.slMesh);
      this.slMesh.geometry.dispose();
    }
    const geo = new THREE.InstancedBufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array((P - 1) * 6 * 3), 3));
    geo.instanceCount = S;
    const m = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader: STREAM_VS, fragmentShader: RIBBON_FS, ...PREMUL, depthTest: false,
      uniforms: { viewProj: { value: new THREE.Matrix4() }, viewport: { value: new THREE.Vector2() }, PTS: { value: this.wrap(t) }, P: { value: P }, width: { value: 1.6 }, alpha: { value: 0.9 }, U: { value: 0.05 } },
    });
    this.slMesh = new THREE.Mesh(geo, m);
    this.slMesh.frustumCulled = false;
    this.slMesh.renderOrder = 7;
    this.scene.add(this.slMesh);
  }

  /** Raw-GL tracer passes, then the Three.js scene. Call after the solver's step/post. */
  render(rs: RenderState) {
    const s = this.solver;
    if (!s) return;
    const gl = this.gl;
    const { nx, ny } = s.dims;
    const three = this.three;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    three.setPixelRatio(dpr);
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    const sz = new THREE.Vector2();
    three.getSize(sz);
    if (sz.x !== w || sz.y !== h) three.setSize(w, h, false);
    const [pw, ph] = this.pixelSize();

    gl.bindVertexArray(this.vao);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.SCISSOR_TEST);
    const P = rs.particles;
    if (P.enabled) {
      this.ensureTracers(P.count, P.trail);
      const fill = this.needFill;
      const src = this.pState![this.pCur];
      const dst = this.pCur ^ 1;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.pFbo![dst]);
      gl.viewport(0, 0, this.pW, this.pH);
      const A = this.advect.use();
      A.tex('STATE', 0, src.tex).tex('VEL', 1, s.velTex.tex).tex('FLAGS', 2, s.flagsTex.tex)
        .i2('N', nx, ny).i('W', this.pW).i('count', P.count).i('emitter', P.emitter === 'rake' ? 1 : 0).i('nozzles', P.nozzles)
        .i('fill', fill ? 1 : 0).i('advance', P.advance ? 1 : 0).f('steps', P.steps).f('maxAge', P.maxAge).f('trail', P.trail)
        .f2('rakeA', rs.rake.a[0], rs.rake.a[1]).f2('rakeB', rs.rake.b[0], rs.rake.b[1]);
      gl.uniform1ui(A.u('seed'), this.seed++);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      this.pCur = dst;
      // write into the history ring
      const C = this.copy.use();
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.pHist!.fbo);
      const rows = fill ? P.trail : 1;
      if (P.advance && !fill) this.head = (this.head + 1) % P.trail;
      for (let r = 0; r < rows; r++) {
        const slot = fill ? r : this.head;
        gl.viewport(0, slot * this.pH, this.pW, this.pH);
        C.tex('SRC', 0, this.pState![dst].tex).i('yOff', slot * this.pH);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
      this.needFill = false;
    }
    const SL = rs.streamlines;
    if (SL.enabled) {
      this.ensureStreamlines(Math.min(SL.points, 2048), SL.seeds);
      const t = this.slTex!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
      gl.viewport(0, 0, t.P, t.S);
      const S = this.stream.use();
      S.tex('VEL', 0, s.velTex.tex).tex('MEAN', 1, s.meanTexture.tex).tex('FLAGS', 2, s.flagsTex.tex).i2('N', nx, ny).i('seeds', t.S)
        .i('useMean', SL.useMean ? 1 : 0).f2('rakeA', rs.rake.a[0], rs.rake.a[1]).f2('rakeB', rs.rake.b[0], rs.rake.b[1]).f('h', SL.step).f('U', Math.max(rs.flow.U, 1e-4));
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    // probes: 8×2 texels (row 0 instantaneous, row 1 mean), read back every few frames
    const np = Math.min(rs.probes.length, 8);
    if (np && this.probeFrame++ % 3 === 0) {
      if (!this.probeTex) {
        const t = makeTex(gl, 8, 2, gl.RGBA32F, gl.RGBA, gl.FLOAT);
        const f = gl.createFramebuffer()!;
        gl.bindFramebuffer(gl.FRAMEBUFFER, f);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t.tex, 0);
        this.probeTex = { tex: t, fbo: f };
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.probeTex.fbo);
      gl.viewport(0, 0, 8, 2);
      const PP = this.probePass.use();
      PP.tex('VEL', 0, s.velTex.tex).tex('MEAN', 1, s.meanTexture.tex).f2('N', nx, ny);
      const pts = new Float32Array(16);
      rs.probes.slice(0, 8).forEach((p, i) => pts.set([p.pos[0], p.pos[1]], 2 * i));
      gl.uniform2fv(PP.u('P[0]'), pts);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      const d = new Float32Array(64);
      gl.readPixels(0, 0, 8, 2, gl.RGBA, gl.FLOAT, d);
      // GL layout is (ux, uy, ρ, ·) → normalise to (ux, uy, uz, ρ)
      this.probeData = Array.from({ length: np }, (_, i) => ({
        inst: [d[4 * i], d[4 * i + 1], 0, d[4 * i + 2]],
        mean: [d[32 + 4 * i], d[32 + 4 * i + 1], 0, d[32 + 4 * i + 2]],
      }));
    } else if (!np) {
      this.probeData = [];
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindVertexArray(null);
    three.resetState();

    // ---- scene uniforms
    const VP = new THREE.Matrix4().fromArray(rs.viewProj);
    const N = new THREE.Vector2(nx, ny);
    const vel = this.wrap(s.velTex), vort = this.wrap(s.vortTex), mean = this.wrap(s.meanTexture);
    const su = this.slice.material.uniforms;
    su.viewProj.value.copy(VP);
    su.VEL.value = vel; su.VORT.value = vort; su.MEAN.value = mean; su.N.value.copy(N);
    su.U.value = rs.flow.U; su.Lref.value = rs.flow.Lref; su.rhoRef.value = rs.flow.rhoRef;
    su.field.value = rs.slice.field; su.cmap.value = rs.slice.cmap;
    su.flags.value = (rs.slice.mean ? 1 : 0) | (rs.slice.recirc ? 2 : 0) | (rs.slice.qContour ? 4 : 0);
    su.vmin.value = rs.slice.vmin; su.vmax.value = rs.slice.vmax; su.opacity.value = rs.slice.opacity; su.qThr.value = rs.slice.qThr;
    const tr = rs.slice.opacity < 0.999;
    Object.assign(this.slice.material, tr ? PREMUL : { blending: THREE.NoBlending, transparent: false, depthWrite: true });
    this.slice.visible = rs.slice.visible;
    const gu = this.ground.material.uniforms;
    gu.viewProj.value.copy(VP); gu.N.value.copy(N); gu.moving.value = rs.ground.moving ? 1 : 0; gu.offset.value = rs.groundOffset;
    this.box.visible = rs.box;
    // box/rake use Three's own camera path: give it the combined matrix via projectionMatrix
    this.cam.projectionMatrix.copy(VP);
    this.cam.projectionMatrixInverse.copy(VP).invert();
    this.cam.matrixWorldInverse.identity();
    this.cam.matrixWorld.identity();
    if (this.body) {
      const mu = this.meshMat.uniforms;
      mu.viewProj.value.copy(VP);
      mu.model.value.fromArray(rs.mesh.model);
      mu.normalM.value.fromArray(rs.mesh.normalMatrix);
      mu.VEL.value = vel; mu.MEAN.value = mean; mu.N.value.copy(N); mu.U.value = rs.flow.U; mu.rhoRef.value = rs.flow.rhoRef;
      mu.eye.value.set(...rs.eye); mu.mode.value = rs.mesh.mode === 'cp' ? 1 : 0; mu.useMean.value = rs.mesh.useMean ? 1 : 0;
      mu.cpMin.value = rs.mesh.cpMin; mu.cpMax.value = rs.mesh.cpMax; mu.opacity.value = rs.mesh.opacity;
      Object.assign(this.meshMat, rs.mesh.opacity < 0.999 ? PREMUL : { blending: THREE.NoBlending, transparent: false, depthWrite: true });
      this.body.visible = rs.mesh.visible;
    }
    if (this.trails) {
      this.trails.visible = P.enabled;
      const tu = this.trails.material.uniforms;
      tu.viewProj.value.copy(VP); tu.viewport.value.set(pw, ph);
      tu.HIST.value = this.wrap(this.pHist!.tex); tu.STATE.value = this.wrap(this.pState![this.pCur]); tu.VEL.value = vel;
      tu.head.value = this.head; tu.colorMode.value = P.colorMode === 'speed' ? 1 : 0; tu.width.value = P.width * dpr; tu.alpha.value = P.alpha;
      tu.N.value.copy(N); tu.U.value = Math.max(rs.flow.U, 1e-5);
    }
    if (this.slMesh) {
      this.slMesh.visible = SL.enabled;
      const u = this.slMesh.material.uniforms;
      u.viewProj.value.copy(VP); u.viewport.value.set(pw, ph); u.width.value = SL.width * dpr; u.alpha.value = SL.alpha; u.U.value = Math.max(rs.flow.U, 1e-5);
    }
    this.rakeLine.visible = this.rakeHandle.visible = rs.rake.visible;
    if (rs.rake.visible) {
      const a = new THREE.Vector3(...rs.rake.a), b = new THREE.Vector3(...rs.rake.b);
      (this.rakeLine.geometry as THREE.BufferGeometry).setFromPoints([a, b]);
      const m = a.clone().add(b).multiplyScalar(0.5);
      const rad = Math.max(ny * 0.012, 2);
      this.rakeHandle.matrix.makeTranslation(m.x, m.y, 0.6).multiply(new THREE.Matrix4().makeScale(rad, rad, 1));
      (this.rakeHandle.material as THREE.MeshBasicMaterial).color.set(rs.rake.active ? 0xffe08a : 0xffc233);
    }
    {
      const pos: number[] = [], col: number[] = [];
      const k = Math.max(nx * 0.008, 1.5);
      for (const p of rs.probes) {
        const [x, y] = p.pos;
        pos.push(x - k, y, 0.6, x + k, y, 0.6, x, y - k, 0.6, x, y + k, 0.6);
        for (let j = 0; j < 4; j++) col.push(...p.color);
      }
      const g = this.probeMarks.geometry;
      g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
      this.probeMarks.visible = pos.length > 0;
    }
    three.render(this.scene, this.cam);
  }
}
