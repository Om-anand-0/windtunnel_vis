import type { App } from '../app';
import { formatRe } from '../analysis/units';
import { colormapCSS } from '../render/colormaps';
import { FIELDS } from '../state';
import { LineChart } from './chart';
import { HUD_TIPS } from './tips';
import { h } from './widgets';

/** Top status bar, right-hand aero readouts with the Cd chart, and the colour legends. */
export class Hud {
  private el: Record<string, HTMLElement> = {};
  private chart: LineChart;
  private legendField: HTMLElement;
  private legendSurface: HTMLElement;

  constructor(private app: App, root: HTMLElement, top: HTMLElement) {
    const kv = (label: string, key: string, cls = '') => {
      const v = h('span', { class: 'v' }, '—');
      this.el[key] = v;
      return h('div', { class: `kv ${cls}` }, h('span', { class: 'k' }, label), v);
    };
    top.innerHTML = '';
    top.append(
      h('div', { class: 'brand' }, h('span', { class: 'logo' }), h('b', {}, 'WINDTUNNEL'), h('span', { class: 'dim' }, ' · lattice Boltzmann')),
      kv('backend', 'backend'),
      kv('grid', 'grid'),
      kv('fps', 'fps'),
      kv('steps/frame', 'spf'),
      kv('MLUPS', 'mlups'),
      kv('t·U/L', 'tconv'),
      h('div', { class: 'spacer' }),
      kv('', 'status', 'status'),
    );

    root.innerHTML = '';
    const big = (label: string, key: string, sub: string) => {
      const v = h('div', { class: 'bigv' }, '—');
      const sb = h('div', { class: 'bigs' }, sub);
      this.el[key] = v;
      this.el[key + 'Sub'] = sb;
      return h('div', { class: 'big' }, h('div', { class: 'bigl' }, label), v, sb);
    };
    root.append(
      h('div', { class: 'hud-title' }, 'Aerodynamics'),
      h('div', { class: 'bigrow' }, big('C_D', 'cd', 'drag'), big('C_L', 'cl', 'lift'), big('C_S', 'cs', 'side')),
    );
    const chartBox = h('div', { class: 'chartbox' }, h('div', { class: 'chart-t' }, h('span', {}, 'C_D history'), h('span', { class: 'dim' }, 'avg ─ inst ···')));
    root.append(chartBox);
    this.chart = new LineChart(chartBox, 96);
    this.chart.xLabel = 't·U/L';
    root.append(
      h('div', { class: 'kvs' },
        kv('Drag force', 'drag'),
        kv('Downforce', 'down'),
        kv('Frontal area', 'area'),
        kv('Blockage', 'block'),
        kv('Re (real)', 'reReal'),
        kv('Re (simulated)', 'reSim'),
        kv('Mach (inlet / max)', 'mach'),
        kv('τ (relaxation)', 'tau'),
        kv('Lattice U', 'U'),
        kv('Δx · Δt', 'dxdt'),
      ),
    );
    for (const [k, tip] of Object.entries(HUD_TIPS)) {
      const host = this.el[k]?.closest('.kv, .big') as HTMLElement | null;
      if (host) host.dataset.tip = tip;
    }
    this.legendField = h('div', { class: 'legend' });
    this.legendSurface = h('div', { class: 'legend' });
    document.getElementById('legends')!.append(this.legendField, this.legendSurface);
  }

  private legend(el: HTMLElement, title: string, unit: string, cmap: Parameters<typeof colormapCSS>[0], lo: number, hi: number, visible: boolean) {
    el.style.display = visible ? '' : 'none';
    if (!visible) return;
    const key = `${title}|${unit}|${cmap}|${lo}|${hi}`;
    if (el.dataset.key === key) return;
    el.dataset.key = key;
    const fmt = (v: number) => (Math.abs(v) >= 10 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2));
    const mid = (lo + hi) / 2;
    el.innerHTML = '';
    el.append(
      h('div', { class: 'lg-t' }, h('b', {}, title), h('span', { class: 'dim' }, ' ' + unit)),
      h('div', { class: 'lg-bar', style: `background:${colormapCSS(cmap)}` }),
      h('div', { class: 'lg-ticks' }, h('span', {}, fmt(lo)), h('span', {}, fmt(mid)), h('span', {}, fmt(hi))),
    );
  }

  update() {
    const a = this.app;
    const r = a.readout();
    const s = a.s;
    const d = a.dims;
    const e = this.el;
    const cells = d.nx * d.ny * d.nz;
    e.backend.textContent = a.backend.kind === 'webgpu' ? 'WebGPU' : 'WebGL2 (fallback)';
    e.backend.title = a.backend.label;
    e.grid.textContent = `${s.mode === '3d' ? 'D3Q19' : 'D2Q9'} ${d.nx}×${d.ny}${d.nz > 1 ? '×' + d.nz : ''} · ${(cells / 1e6).toFixed(2)} M`;
    e.fps.textContent = r.fps.toFixed(0);
    e.spf.textContent = String(r.stepsPerFrame);
    e.mlups.textContent = r.mlups.toFixed(0);
    e.tconv.textContent = r.convTime.toFixed(1);
    const stab = a.stabilityEvents;
    e.status.textContent = s.paused ? 'PAUSED' : stab ? `RUNNING · stabilised ×${stab}` : 'RUNNING';
    e.status.className = 'v ' + (s.paused ? 'paused' : stab ? 'warn' : 'ok');

    const f3 = (v: number) => (isFinite(v) ? v.toFixed(3) : '—');
    e.cd.textContent = f3(r.cd);
    e.cl.textContent = f3(r.cl);
    e.cs.textContent = s.mode === '3d' ? f3(r.cs) : 'n/a';
    e.cdSub.textContent = s.mode === '3d' ? 'drag' : 'drag (2D, per span)';
    const speed = s.speedKmh / 3.6;
    const q = 0.5 * 1.225 * speed * speed * r.frontalM2;
    e.drag.textContent = isFinite(r.dragN) ? `${r.dragN.toFixed(0)} N @ ${s.speedKmh.toFixed(0)} km/h` : '—';
    e.down.textContent = isFinite(r.cl) ? `${(-r.cl * q).toFixed(0)} N` : '—';
    e.area.textContent = `${r.frontalM2.toFixed(2)} m²`;
    e.block.textContent = `${(r.blockage * 100).toFixed(1)} %`;
    e.block.classList.toggle('warnv', r.blockage > 0.15);
    const settling = r.convTime < 1;
    if (settling) {
      e.cdSub.textContent = `settling… t·U/L ${r.convTime.toFixed(2)} / 1`;
    }
    e.reReal.textContent = formatRe(r.flow.reReal);
    e.reSim.textContent = formatRe(r.flow.reSim) + (s.reOverrideOn ? ' (override)' : '');
    e.mach.textContent = `${r.flow.mach.toFixed(3)} / ${r.maxMach.toFixed(3)}`;
    e.mach.classList.toggle('warnv', r.maxMach > 0.4);
    e.tau.textContent = r.flow.tau.toFixed(5);
    e.U.textContent = `${a.currentU.toFixed(4)} cells/step`;
    e.dxdt.textContent = `${(r.flow.dx * 1000).toFixed(1)} mm · ${(r.flow.dt * 1e6).toFixed(1)} µs`;

    // once past the start-up transient, hide it so the y-range follows the settled signal
    const t0 = a.cdAvgSeries.t.length > 3 ? 1 : -Infinity;
    const keep = (sr: { t: number[]; v: number[] }) => {
      const i = sr.t.findIndex((t) => t >= t0);
      return i <= 0 ? sr : { t: sr.t.slice(i), v: sr.v.slice(i) };
    };
    const inst = keep(a.cdSeries);
    this.chart.draw([
      { t: inst.t, v: inst.v, color: '#6f86a8', width: 1, alpha: 0.55 },
      { t: a.cdAvgSeries.t, v: a.cdAvgSeries.v, color: '#ffd166', width: 1.8 },
    ]);

    const fd = FIELDS[s.field];
    this.legend(this.legendField, (s.sliceMean ? 'Mean ' : '') + fd.label, fd.unit, s.cmap, s.vmin, s.vmax, s.sliceOn);
    this.legend(this.legendSurface, 'Surface Cp', s.surfaceMean ? '(time-avg)' : '', 'turbo', -1.5, 1, s.bodyOn && s.surface === 'cp');
  }
}
