import { initWebGPU } from '../gpu/device';
import { CASES } from '../analysis/validation';
import { SolverGPU } from '../solver/SolverGPU';
import { VoxelizerGPU } from '../voxelize/VoxelizerGPU';
import { buildPreset } from '../voxelize/presets';
import { placementMatrix } from '../voxelize/mesh';

const out = document.getElementById('out')!;
const log = (s: string) => { out.textContent += '\n' + s; console.log(s); };

async function main() {
  const g = await initWebGPU();
  if (!g) { log('NO WEBGPU'); (window as any).__done = true; return; }
  log('adapter: ' + g.adapterInfo + ' maxBinding=' + g.maxBinding);
  const params = new URLSearchParams(location.search);
  const which = (params.get('case') ?? 'freestream').split(',');
  const results: any[] = [];
  if (which.includes('vox')) {
    const vox = new VoxelizerGPU(g.device);
    for (const [id, dims] of [['sphere', { nx: 128, ny: 64, nz: 64 }], ['sphere', { nx: 512, ny: 192, nz: 1 }], ['sedan', { nx: 192, ny: 96, nz: 96 }], ['f1', { nx: 1024, ny: 384, nz: 1 }], ['truck', { nx: 192, ny: 96, nz: 96 }]] as const) {
      const s = new SolverGPU(g.device, dims, { u: 0.05, nu: 0.01, cs: 0.1, ground: 'noslip', sides: 'freeslip', spongeNu: 0, spongeStart: 0.9, emaAlpha: 0 });
      const mesh = buildPreset(id);
      const pl = placementMatrix(mesh, dims, { mode: id === 'sphere' ? 'center' : 'ground', lengthFrac: 1 / 3, diamFrac: 0.3, xFrac: 0.35, yawDeg: 0, pitchDeg: 0, rideCells: 0 });
      const t0 = performance.now();
      const info = await vox.voxelize(s, mesh, pl.matrix.elements);
      const dt = performance.now() - t0;
      let expect = '';
      if (id === 'sphere') {
        const r = (0.3 * dims.ny) / 2;
        expect = dims.nz > 1 ? ` expected vol ${(4 / 3 * Math.PI * r ** 3).toFixed(0)} area ${(Math.PI * r * r).toFixed(0)}` : ` expected area ${(Math.PI * r * r).toFixed(0)} height ${(2 * r).toFixed(1)}`;
      }
      log(`vox ${id} ${dims.nx}x${dims.ny}x${dims.nz}: solid=${info.solidCells} frontal=${info.frontal} bbox=${info.min}..${info.max} (${dt.toFixed(0)} ms)${expect}`);
      if (dims.nz === 1) {
        const f = await s.readFlags();
        // ascii dump downsampled
        const sx = Math.ceil(dims.nx / 128), sy = Math.ceil(dims.ny / 48);
        let art = '';
        for (let y = dims.ny - 1; y >= 0; y -= sy) {
          let line = '';
          for (let x = 0; x < dims.nx; x += sx) line += f[x + dims.nx * y] ? '#' : '.';
          art += line + '\n';
        }
        log(art);
      }
      results.push({ id, info });
      s.destroy();
    }
  }
  if (which.includes('ground')) {
    for (const ground of ['moving', 'noslip', 'freeslip'] as const) {
      const dims = { nx: 256, ny: 64, nz: 1 };
      const U = 0.06;
      const s = new SolverGPU(g.device, dims, { u: U, nu: 0.005, cs: 0.14, ground, sides: 'freeslip', spongeNu: 0.04, spongeStart: 0.84, emaAlpha: 0 });
      s.uploadFlags(new Uint32Array(dims.nx * dims.ny));
      await s.stepAndSample(3000);
      const f = await s.readPopulations();
      const n = s.n;
      const c = s.lattice.c;
      const prof = (x: number) => [0, 1, 2, 4, 8, 16, 32, 63].map((y) => {
        const idx = x + dims.nx * y;
        let r = 0, ux = 0, uy = 0;
        for (let i = 0; i < 9; i++) { const v = f[i * n + idx]; r += v; ux += v * c[i][0]; uy += v * c[i][1]; }
        return `${y}:${(ux / r / U).toFixed(3)}/${(uy / r / U).toFixed(3)}`;
      }).join(' ');
      log(`${ground} x=20  ${prof(20)}`);
      log(`${ground} x=128 ${prof(128)}`);
      s.destroy();
    }
  }
  if (which.includes('car')) {
    const vox = new VoxelizerGPU(g.device);
    const id = params.get('vehicle') ?? 'sedan';
    const q = params.get('dims') ?? '512,192,1';
    const [nx, ny, nz] = q.split(',').map(Number);
    const dims = { nx, ny, nz };
    const U = parseFloat(params.get('u') ?? '0.061');
    const Re = parseFloat(params.get('re') ?? '20000');
    const L = nx / 3;
    const nu = (U * L) / Re;
    const cs = parseFloat(params.get('cs') ?? '0.14');
    const s = new SolverGPU(g.device, dims, { u: U, nu, cs, ground: (params.get('ground') ?? 'moving') as any, sides: 'freeslip', spongeNu: 0.04, spongeStart: 0.84, emaAlpha: 0, collision: (params.get('coll') ?? 'regularized') as any, tauWall: parseFloat(params.get('tw') ?? '0.53'), spongeIn: parseFloat(params.get('sin') ?? '0') });
    const mesh = buildPreset(id);
    const pl = placementMatrix(mesh, dims, { mode: id === 'sphere' || id === 'cylinder' ? 'center' : 'ground', lengthFrac: 1 / 3, diamFrac: 0.3, xFrac: 0.34, yawDeg: 0, pitchDeg: 0, rideCells: parseFloat(params.get('ride') ?? '0') });
    const info = await vox.voxelize(s, mesh, pl.matrix.elements);
    log(`car ${id} dims ${q} frontal ${info.frontal} tau ${(3 * nu + 0.5).toFixed(5)} Re ${Re}`);
    const total = parseInt(params.get('steps') ?? '20000');
    const chunk = 250;
    let acc = 0, n = 0;
    for (let st = 0; st < total; st += chunk) {
      const r = await s.stepAndSample(chunk);
      const cd = r.fx / (0.5 * U * U * info.frontal);
      const cl = r.fy / (0.5 * U * U * info.frontal);
      if (st > total / 2) { acc += cd; n++; }
      if ((st / chunk) % 8 === 0 || r.unstable) log(`step ${r.step} t*=${((r.step * U) / L).toFixed(2)} cd=${cd.toFixed(3)} cl=${cl.toFixed(3)} maxMa=${(r.maxU * Math.sqrt(3)).toFixed(3)} rhoRef=${r.rhoRef.toFixed(4)} ${r.unstable ? 'UNSTABLE' : ''}`);
      if (r.unstable) break;
    }
    log(`mean cd (second half) ${(acc / n).toFixed(3)}`);
    {
      const f = await s.readPopulations();
      const c = s.lattice.c;
      const zc = Math.floor(nz / 2);
      for (const x of [2, 10, 30, 60, 100, 160].map((v) => Math.round((v * nx) / 512))) {
        const row = [0, 1, 2, 4, 8, 16, 32, 64, 128, 190].filter((y) => y < ny).map((y) => {
          const idx = x + nx * (y + ny * zc);
          let r = 0, ux = 0;
          for (let i = 0; i < s.lattice.q; i++) { const v = f[i * s.n + idx]; r += v; ux += v * c[i][0]; }
          return `${y}:${(ux / r / U).toFixed(2)}|${r.toFixed(3)}`;
        }).join(' ');
        log(`x=${x} ${row}`);
      }
    }
    s.destroy();
  }
  for (const c of CASES) {
    if (!which.includes(c.id)) continue;
    log('running ' + c.name);
    try {
      const r = await c.run((d, p) => new SolverGPU(g.device, d, p), (f, m) => console.log(c.id, (f * 100).toFixed(0) + '%', m ?? ''), () => false);
      log(JSON.stringify({ ...r, series: undefined }, null, 1));
      if (r.series) log('series tail: ' + r.series.v.slice(-40).map((v) => v.toFixed(3)).join(' '));
      results.push(r);
    } catch (e) {
      log('ERROR ' + e);
    }
  }
  (window as any).__results = results;
  (window as any).__done = true;
  log('DONE');
}
main();
