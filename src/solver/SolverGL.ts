import { D2Q9, fl } from './lattice';
import type { GridDims, SolverParams, SolverStats } from './types';

/**
 * WebGL2 fragment-shader D2Q9 solver (fallback when WebGPU is unavailable).
 * Populations live in three RGBA32F textures (f0‥f3, f4‥f7, f8) written with MRT and ping-ponged.
 * Physics and boundary conditions mirror the WGSL kernel (regularized BGK + Smagorinsky in the bulk,
 * BGK at walls, moving belt with contact patches, inlet absorbing layer, outlet sponge).
 */
const L = D2Q9;
const arr = (v: number[], t = 'float') => `${t}[9](${v.map((x) => (t === 'float' ? fl(x) : String(x))).join(', ')})`;

const VS = /* glsl */ `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const HEAD = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
const ivec2 C[9] = ivec2[9](${L.c.map((c) => `ivec2(${c[0]}, ${c[1]})`).join(', ')});
const float W[9] = ${arr(L.w)};
const int OPP[9] = ${arr(L.opp, 'int')};
const int MY[9] = ${arr(L.mirrorY, 'int')};
uniform sampler2D F0;
uniform sampler2D F1;
uniform sampler2D F2;
uniform sampler2D FLAGS;
uniform ivec2 N;
float getF(int i, ivec2 p) {
  if (i < 4) return texelFetch(F0, p, 0)[i];
  if (i < 8) return texelFetch(F1, p, 0)[i - 4];
  return texelFetch(F2, p, 0).x;
}
bool solid(ivec2 p) {
  if (p.x < 0 || p.y < 0 || p.x >= N.x || p.y >= N.y) return false;
  return texelFetch(FLAGS, p, 0).r > 0.5;
}
`;

const STREAM_COLLIDE = /* glsl */ `${HEAD}
uniform float tau0, csmag2, U, spongeStart, spongeNu, uGround, tauWall, spongeIn, contactH;
uniform int ground, collision;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  int ix = p.x, iy = p.y;
  if (solid(p)) { o0 = vec4(W[0], W[1], W[2], W[3]); o1 = vec4(W[4], W[5], W[6], W[7]); o2 = vec4(W[8], 0.0, 0.0, 0.0); return; }
  vec4 L0 = texelFetch(F0, p, 0), L1 = texelFetch(F1, p, 0), L2 = texelFetch(F2, p, 0);
  float loc[9] = float[9](L0.x, L0.y, L0.z, L0.w, L1.x, L1.y, L1.z, L1.w, L2.x);
  bool wall = false, gwall = false;
  float belt = ground == 2 ? 1.0 : 0.0;
  if (belt > 0.0 && iy == 0) {
    int r = int(contactH);
    for (int dx = -3; dx <= 3; dx++) {
      if (abs(dx) > r) continue;
      if (solid(ivec2(ix + dx, 0)) || solid(ivec2(ix + dx, 1))) belt = 0.0;
    }
  }
  float f[9];
  f[0] = loc[0];
  for (int i = 1; i < 9; i++) {
    ivec2 c = C[i];
    int sx = ix - c.x, sy = iy - c.y;
    if (sx < 0) {
      float cu = float(c.x) * U;
      f[i] = W[i] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * U * U);
      continue;
    }
    sx = min(sx, N.x - 1);
    if (sy < 0) {
      if (ground != 0) { wall = true; gwall = true; f[i] = loc[OPP[i]] + belt * 6.0 * W[i] * float(c.x) * uGround; }
      else f[i] = getF(MY[i], ivec2(sx, iy));
      continue;
    }
    if (sy >= N.y) { f[i] = getF(MY[i], ivec2(sx, iy)); continue; }
    ivec2 s = ivec2(sx, sy);
    if (solid(s)) { wall = true; f[i] = loc[OPP[i]]; }
    else f[i] = getF(i, s);
  }
  float rho = 0.0, jx = 0.0, jy = 0.0, pxx = 0.0, pxy = 0.0, pyy = 0.0;
  for (int i = 0; i < 9; i++) {
    vec2 c = vec2(C[i]);
    rho += f[i]; jx += f[i] * c.x; jy += f[i] * c.y;
    pxx += f[i] * c.x * c.x; pxy += f[i] * c.x * c.y; pyy += f[i] * c.y * c.y;
  }
  float ux = jx / rho, uy = jy / rho, usq = ux * ux + uy * uy;
  pxx -= rho * (ux * ux + 1.0 / 3.0);
  pxy -= rho * ux * uy;
  pyy -= rho * (uy * uy + 1.0 / 3.0);
  float piN = sqrt(pxx * pxx + 2.0 * pxy * pxy + pyy * pyy);
  float fx = float(ix);
  float sp = clamp((fx - spongeStart) / max(float(N.x) - spongeStart, 1.0), 0.0, 1.0);
  float t0 = tau0 + 3.0 * spongeNu * sp * sp;
  if (gwall) t0 = max(t0, tauWall);
  float tau = 0.5 * (t0 + sqrt(t0 * t0 + ${fl(18 * Math.SQRT2)} * csmag2 * piN / rho));
  float k1 = (1.0 - 1.0 / tau);
  bool bgk = wall || collision == 1;
  float si = clamp(1.0 - fx / max(spongeIn, 1.0), 0.0, 1.0);
  float sIn = spongeIn > 0.0 ? 0.08 * si * si : 0.0;
  float trP = (pxx + pyy) / 3.0;
  float o[9];
  for (int i = 0; i < 9; i++) {
    vec2 c = vec2(C[i]);
    float cu = c.x * ux + c.y * uy;
    float feq = W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq);
    float qp = c.x * c.x * pxx + 2.0 * c.x * c.y * pxy + c.y * c.y * pyy - trP;
    float fo = feq + (bgk ? k1 * (f[i] - feq) : W[i] * 4.5 * k1 * qp);
    if (sIn > 0.0) {
      float cuI = c.x * U;
      fo = mix(fo, W[i] * (1.0 + 3.0 * cuI + 4.5 * cuI * cuI - 1.5 * U * U), sIn);
    }
    o[i] = fo;
  }
  o0 = vec4(o[0], o[1], o[2], o[3]);
  o1 = vec4(o[4], o[5], o[6], o[7]);
  o2 = vec4(o[8], 0.0, 0.0, 0.0);
}`;

const INIT = /* glsl */ `${HEAD}
uniform float U;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  float u = solid(p) ? 0.0 : U;
  float o[9];
  for (int i = 0; i < 9; i++) {
    float cu = float(C[i].x) * u;
    o[i] = W[i] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u * u);
  }
  o0 = vec4(o[0], o[1], o[2], o[3]);
  o1 = vec4(o[4], o[5], o[6], o[7]);
  o2 = vec4(o[8], 0.0, 0.0, 0.0);
}`;

/** vel = (ux, uy, ρ, solid); stat = (Fx, Fy, |u|², ρ on the reference column); mean = EMA(ux, uy, ρ, |u|²) */
const MACRO = /* glsl */ `${HEAD}
uniform sampler2D MEAN;
uniform float alpha;
uniform int refX;
layout(location = 0) out vec4 vel;
layout(location = 1) out vec4 stat;
layout(location = 2) out vec4 mean;
float rhoAt(ivec2 q) { float r = 0.0; for (int i = 0; i < 9; i++) r += getF(i, q); return r; }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 m = texelFetch(MEAN, p, 0);
  if (solid(p)) {
    float rs = 0.0, rn = 0.0;
    for (int k = 1; k < 5; k++) {
      ivec2 q = p + C[k];
      if (q.x >= 0 && q.y >= 0 && q.x < N.x && q.y < N.y && !solid(q)) { rs += rhoAt(q); rn += 1.0; }
    }
    float r = rn > 0.0 ? rs / rn : 1.0;
    vel = vec4(0.0, 0.0, r, 1.0);
    stat = vec4(0.0);
    mean = mix(m, vec4(0.0, 0.0, r, 0.0), alpha);
    return;
  }
  float f[9];
  float rho = 0.0, jx = 0.0, jy = 0.0;
  for (int i = 0; i < 9; i++) { f[i] = getF(i, p); rho += f[i]; jx += f[i] * float(C[i].x); jy += f[i] * float(C[i].y); }
  float ux = jx / rho, uy = jy / rho, uu = ux * ux + uy * uy;
  vec2 F = vec2(0.0);
  for (int i = 1; i < 9; i++) {
    if (solid(p + C[i])) F += 2.0 * (f[i] - W[i]) * vec2(C[i]);
  }
  bool bad = !(uu < 1.0e6) || !(rho > 0.0);
  vel = vec4(ux, uy, rho, 0.0);
  stat = vec4(F, bad ? 1.0e6 : uu, p.x == refX ? rho : 0.0);
  mean = mix(m, vec4(ux, uy, rho, uu), alpha);
}`;

/** vort = (ω_z, Q, TI, 0) */
const DERIVED = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D VEL;
uniform sampler2D MEAN;
uniform ivec2 N;
uniform float U;
out vec4 vort;
vec2 v(ivec2 q) { return texelFetch(VEL, clamp(q, ivec2(0), N - 1), 0).xy; }
bool sol(ivec2 q) { return texelFetch(VEL, clamp(q, ivec2(0), N - 1), 0).w > 0.5; }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 me = texelFetch(VEL, p, 0);
  vec4 m = texelFetch(MEAN, p, 0);
  if (me.w > 0.5) { vort = vec4(0.0); return; }
  vec2 dx = 0.5 * (v(p + ivec2(1, 0)) - v(p - ivec2(1, 0)));
  vec2 dy = 0.5 * (v(p + ivec2(0, 1)) - v(p - ivec2(0, 1)));
  float wz = dx.y - dy.x;
  float sxx = dx.x, syy = dy.y, sxy = 0.5 * (dx.y + dy.x);
  float S2 = sxx * sxx + syy * syy + 2.0 * sxy * sxy;
  float W2 = 0.5 * wz * wz;
  float k = max(m.w - dot(m.xy, m.xy), 0.0);
  float near = 0.0;
  for (int d = 1; d <= 2; d++) {
    float w = d == 1 ? 1.0 : 0.5;
    if (sol(p + ivec2(d, 0)) || sol(p - ivec2(d, 0)) || sol(p + ivec2(0, d)) || sol(p - ivec2(0, d))) near = max(near, w);
  }
  vort = vec4(wz, 0.5 * (W2 - S2) * (1.0 - near), sqrt(k / 3.0) / max(U, 1e-6), 0.0);
}`;

/** 8×8 block reduction: sum x, y, w; max z. */
const REDUCE = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D SRC;
uniform ivec2 SN;
out vec4 o;
void main() {
  ivec2 b = ivec2(gl_FragCoord.xy) * 8;
  vec4 acc = vec4(0.0);
  for (int j = 0; j < 8; j++) for (int i = 0; i < 8; i++) {
    ivec2 q = b + ivec2(i, j);
    if (q.x >= SN.x || q.y >= SN.y) continue;
    vec4 s = texelFetch(SRC, q, 0);
    acc.xyw += s.xyw;
    acc.z = max(acc.z, s.z);
  }
  o = acc;
}`;

export interface GLTex {
  tex: WebGLTexture;
  w: number;
  h: number;
}

export function compile(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const mk = (type: number, src: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('GLSL compile error: ' + gl.getShaderInfoLog(s) + '\n' + src.split('\n').map((l, i) => `${i + 1}: ${l}`).join('\n'));
    return s;
  };
  const p = gl.createProgram()!;
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('GLSL link error: ' + gl.getProgramInfoLog(p));
  return p;
}

export function makeTex(gl: WebGL2RenderingContext, w: number, h: number, internal: number, format: number, type: number, linear = false, data: ArrayBufferView | null = null): GLTex {
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, data);
  const f = linear ? gl.LINEAR : gl.NEAREST;
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return { tex, w, h };
}

export class FullscreenPass {
  readonly prog: WebGLProgram;
  private loc = new Map<string, WebGLUniformLocation | null>();
  constructor(readonly gl: WebGL2RenderingContext, fs: string) {
    this.prog = compile(gl, VS, fs);
  }
  u(name: string) {
    if (!this.loc.has(name)) this.loc.set(name, this.gl.getUniformLocation(this.prog, name));
    return this.loc.get(name)!;
  }
  use() {
    this.gl.useProgram(this.prog);
    return this;
  }
  tex(name: string, unit: number, t: WebGLTexture) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.uniform1i(this.u(name), unit);
    return this;
  }
  f(name: string, v: number) { this.gl.uniform1f(this.u(name), v); return this; }
  i(name: string, v: number) { this.gl.uniform1i(this.u(name), v); return this; }
  i2(name: string, a: number, b: number) { this.gl.uniform2i(this.u(name), a, b); return this; }
  f2(name: string, a: number, b: number) { this.gl.uniform2f(this.u(name), a, b); return this; }
  f4(name: string, a: number, b: number, c: number, d: number) { this.gl.uniform4f(this.u(name), a, b, c, d); return this; }
}

export class SolverGL {
  readonly dims: GridDims;
  readonly n: number;
  readonly is3D = false;
  stepCount = 0;
  private p: SolverParams;

  private fA: GLTex[];
  private fB: GLTex[];
  private fboA: WebGLFramebuffer;
  private fboB: WebGLFramebuffer;
  flagsTex: GLTex;
  velTex: GLTex;
  vortTex: GLTex;
  private statTex: GLTex;
  private meanTex: [GLTex, GLTex];
  meanCur = 0;
  private macroFbo: [WebGLFramebuffer, WebGLFramebuffer];
  private vortFbo: WebGLFramebuffer;
  private reduceChain: { tex: GLTex; fbo: WebGLFramebuffer }[] = [];
  private scPass: FullscreenPass;
  private initPass: FullscreenPass;
  private macroPass: FullscreenPass;
  private derivedPass: FullscreenPass;
  private reducePass: FullscreenPass;
  private vao: WebGLVertexArrayObject;
  private parity = 0;
  private pendingInit = true;
  private pbo: WebGLBuffer;
  private fence: WebGLSync | null = null;
  private fenceStep = 0;

  constructor(readonly gl: WebGL2RenderingContext, dims: GridDims, params: SolverParams) {
    if (dims.nz !== 1) throw new Error('SolverGL is 2D only');
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('EXT_color_buffer_float is required');
    // float32 textures are only filterable with this extension; otherwise sample them NEAREST
    const f32Linear = !!gl.getExtension('OES_texture_float_linear');
    this.dims = dims;
    this.n = dims.nx * dims.ny;
    this.p = { ...params };
    const { nx, ny } = dims;
    const F = () => makeTex(gl, nx, ny, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    this.fA = [F(), F(), F()];
    this.fB = [F(), F(), F()];
    const mrt = (texs: GLTex[]) => {
      const fbo = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      texs.forEach((t, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t.tex, 0));
      gl.drawBuffers(texs.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
      const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (st !== gl.FRAMEBUFFER_COMPLETE) throw new Error('framebuffer incomplete: 0x' + st.toString(16));
      return fbo;
    };
    this.fboA = mrt(this.fA);
    this.fboB = mrt(this.fB);
    this.flagsTex = makeTex(gl, nx, ny, gl.R8, gl.RED, gl.UNSIGNED_BYTE, false, new Uint8Array(this.n));
    this.velTex = makeTex(gl, nx, ny, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, true);
    this.vortTex = makeTex(gl, nx, ny, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, true);
    this.statTex = makeTex(gl, nx, ny, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    const m0 = new Float32Array(this.n * 4);
    for (let i = 2; i < m0.length; i += 4) m0[i] = 1;
    this.meanTex = [makeTex(gl, nx, ny, gl.RGBA32F, gl.RGBA, gl.FLOAT, f32Linear, m0), makeTex(gl, nx, ny, gl.RGBA32F, gl.RGBA, gl.FLOAT, f32Linear, m0)];
    this.macroFbo = [mrt([this.velTex, this.statTex, this.meanTex[1]]), mrt([this.velTex, this.statTex, this.meanTex[0]])];
    this.vortFbo = mrt([this.vortTex]);
    let w = nx, h = ny;
    while (w > 1 || h > 1) {
      w = Math.ceil(w / 8);
      h = Math.ceil(h / 8);
      const t = makeTex(gl, w, h, gl.RGBA32F, gl.RGBA, gl.FLOAT);
      this.reduceChain.push({ tex: t, fbo: mrt([t]) });
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.scPass = new FullscreenPass(gl, STREAM_COLLIDE);
    this.initPass = new FullscreenPass(gl, INIT);
    this.macroPass = new FullscreenPass(gl, MACRO);
    this.derivedPass = new FullscreenPass(gl, DERIVED);
    this.reducePass = new FullscreenPass(gl, REDUCE);
    this.vao = gl.createVertexArray()!;
    this.pbo = gl.createBuffer()!;
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo);
    gl.bufferData(gl.PIXEL_PACK_BUFFER, 16, gl.STREAM_READ);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
  }

  get params(): SolverParams {
    return this.p;
  }

  setParams(p: Partial<SolverParams>) {
    Object.assign(this.p, p);
  }

  reset() {
    this.pendingInit = true;
    this.stepCount = 0;
    this.clearStats();
  }

  clearStats() {
    const gl = this.gl;
    const m0 = new Float32Array(this.n * 4);
    for (let i = 2; i < m0.length; i += 4) m0[i] = 1;
    for (const t of this.meanTex) {
      gl.bindTexture(gl.TEXTURE_2D, t.tex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.dims.nx, this.dims.ny, gl.RGBA, gl.FLOAT, m0);
    }
  }

  uploadFlags(flags: Uint32Array | Uint8Array) {
    const gl = this.gl;
    const u8 = flags instanceof Uint8Array ? flags : Uint8Array.from(flags, (v) => (v ? 1 : 0));
    const scaled = new Uint8Array(u8.length);
    for (let i = 0; i < u8.length; i++) scaled[i] = u8[i] ? 255 : 0;
    gl.bindTexture(gl.TEXTURE_2D, this.flagsTex.tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.dims.nx, this.dims.ny, gl.RED, gl.UNSIGNED_BYTE, scaled);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
  }

  private bindF(pass: FullscreenPass, src: GLTex[]) {
    pass.tex('F0', 0, src[0].tex).tex('F1', 1, src[1].tex).tex('F2', 2, src[2].tex).tex('FLAGS', 3, this.flagsTex.tex).i2('N', this.dims.nx, this.dims.ny);
  }

  private get cur(): GLTex[] {
    return this.parity === 0 ? this.fA : this.fB;
  }

  /** Run `steps` steps (GL state is left dirty — callers using Three.js must resetState()). */
  step(steps: number) {
    const gl = this.gl;
    const { nx, ny } = this.dims;
    gl.bindVertexArray(this.vao);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.SCISSOR_TEST);
    gl.disable(gl.CULL_FACE);
    gl.colorMask(true, true, true, true);
    gl.viewport(0, 0, nx, ny);
    if (this.pendingInit) {
      this.initPass.use();
      for (const [fbo, src] of [[this.fboA, this.fB], [this.fboB, this.fA]] as const) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        this.bindF(this.initPass, src);
        this.initPass.f('U', this.p.u);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }
      this.parity = 0;
      this.pendingInit = false;
    }
    const P = this.scPass.use();
    const p = this.p;
    P.f('tau0', 3 * p.nu + 0.5).f('csmag2', p.cs * p.cs).f('U', p.u).f('spongeStart', p.spongeStart * nx).f('spongeNu', p.spongeNu)
      .f('uGround', p.ground === 'moving' ? p.u : 0).f('tauWall', p.tauWall ?? 0.5).f('spongeIn', (p.spongeIn ?? 0) * nx).f('contactH', 2)
      .i('ground', p.ground === 'freeslip' ? 0 : p.ground === 'noslip' ? 1 : 2).i('collision', p.collision === 'bgk' ? 1 : 0);
    for (let s = 0; s < steps; s++) {
      const src = this.cur;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.parity === 0 ? this.fboB : this.fboA);
      this.bindF(P, src);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      this.parity ^= 1;
    }
    this.stepCount += steps;
  }

  /** Macroscopic fields, EMA statistics, derived fields and the force reduction. */
  post(derived = true) {
    const gl = this.gl;
    const { nx, ny } = this.dims;
    gl.bindVertexArray(this.vao);
    gl.viewport(0, 0, nx, ny);
    const M = this.macroPass.use();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.macroFbo[this.meanCur]);
    this.bindF(M, this.cur);
    M.tex('MEAN', 4, this.meanTex[this.meanCur].tex).f('alpha', this.p.emaAlpha).i('refX', Math.min(4, nx - 1));
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.meanCur ^= 1;
    if (derived) {
      const D = this.derivedPass.use();
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.vortFbo);
      D.tex('VEL', 0, this.velTex.tex).tex('MEAN', 1, this.meanTex[this.meanCur].tex).i2('N', nx, ny).f('U', Math.max(this.p.u, 1e-4));
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    // reduction chain
    const R = this.reducePass.use();
    let src = this.statTex;
    for (const r of this.reduceChain) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, r.fbo);
      gl.viewport(0, 0, r.tex.w, r.tex.h);
      R.tex('SRC', 0, src.tex).i2('SN', src.w, src.h);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      src = r.tex;
    }
  }

  get meanTexture(): GLTex {
    return this.meanTex[this.meanCur];
  }

  private decode(d: Float32Array): SolverStats {
    const ny = this.dims.ny;
    return {
      fx: d[0], fy: d[1], fz: 0,
      maxU: Math.sqrt(Math.max(d[2], 0)),
      rhoRef: d[3] / ny,
      unstable: !isFinite(d[0]) || d[2] > 0.25,
      step: this.stepCount,
    };
  }

  /** Kick off an async readback of the last reduction (if none is pending) and poll the previous one. */
  pollStats(onStats: (s: SolverStats) => void) {
    const gl = this.gl;
    const last = this.reduceChain[this.reduceChain.length - 1];
    if (this.fence) {
      const st = gl.clientWaitSync(this.fence, 0, 0);
      if (st === gl.TIMEOUT_EXPIRED) return;
      gl.deleteSync(this.fence);
      this.fence = null;
      const d = new Float32Array(4);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo);
      gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, d);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      const s = this.decode(d);
      s.step = this.fenceStep;
      onStats(s);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, last.fbo);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, 0);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    this.fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    this.fenceStep = this.stepCount;
    gl.flush();
  }

  /** Synchronous variant used by validation runs. */
  async stepAndSample(steps: number): Promise<SolverStats> {
    const gl = this.gl;
    this.step(steps);
    this.post(false);
    const last = this.reduceChain[this.reduceChain.length - 1];
    gl.bindFramebuffer(gl.FRAMEBUFFER, last.fbo);
    const d = new Float32Array(4);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, d);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    await new Promise((r) => setTimeout(r, 0));
    return this.decode(d);
  }

  destroy() {
    const gl = this.gl;
    for (const t of [...this.fA, ...this.fB, this.flagsTex, this.velTex, this.vortTex, this.statTex, ...this.meanTex, ...this.reduceChain.map((r) => r.tex)]) gl.deleteTexture(t.tex);
    for (const f of [this.fboA, this.fboB, ...this.macroFbo, this.vortFbo, ...this.reduceChain.map((r) => r.fbo)]) gl.deleteFramebuffer(f);
    gl.deleteBuffer(this.pbo);
    for (const p of [this.scPass, this.initPass, this.macroPass, this.derivedPass, this.reducePass]) gl.deleteProgram(p.prog);
  }
}
