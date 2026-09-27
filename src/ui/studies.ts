import type { App } from '../app';
import { csv, download, stamp } from '../analysis/export';
import { LineChart } from './chart';
import { h, toast } from './widgets';

/** Configuration fields a pinned result remembers (and can restore). */
const CFG_KEYS = ['vehicle', 'lengthM', 'yaw', 'pitch', 'rideMm', 'speedKmh', 'nuAir', 'reOverrideOn', 'reOverride', 'ground', 'lesCs', 'mode', 'quality2D', 'quality3D'] as const;
type Cfg = Pick<App['s'], (typeof CFG_KEYS)[number]>;

interface Pinned {
  label: string;
  cfg: Cfg;
  cd: number;
  cdSE: number;
  cl: number;
  cs: number;
  dragN: number;
  converged: boolean;
  reSim: number;
  thumb: string;
}

type SweepParam = 'yaw' | 'speedKmh' | 'rideMm' | 'pitch';
const PARAMS: Record<SweepParam, { label: string; unit: string; from: number; to: number; n: number }> = {
  yaw: { label: 'Yaw (crosswind)', unit: '°', from: 0, to: 15, n: 4 },
  speedKmh: { label: 'Wind speed', unit: 'km/h', from: 60, to: 240, n: 4 },
  rideMm: { label: 'Ride height offset', unit: 'mm', from: 0, to: 150, n: 4 },
  pitch: { label: 'Pitch', unit: '°', from: -3, to: 3, n: 4 },
};

/** A/B comparison of pinned results and automated parameter sweeps. */
export class Studies {
  private root: HTMLElement;
  private table: HTMLElement;
  private sweepBox: HTMLElement;
  private pinned: Pinned[] = [];
  private sweep: { param: SweepParam; pts: { v: number; cd: number; cdSE: number; cl: number; cs: number }[] } | null = null;
  private running = false;
  private abortFlag = false;

  constructor(private app: App) {
    this.root = h('div', { class: 'modal hidden' });
    const card = h('div', { class: 'modal-card wide' });
    const close = h('button', { class: 'close', title: 'Close' }, '✕');
    close.addEventListener('click', () => this.close());
    this.table = h('div', { class: 'pins' });
    this.sweepBox = h('div', { class: 'sweep' });
    const pin = h('button', { class: 'accent' }, 'Pin current result');
    pin.addEventListener('click', () => this.pinCurrent());
    const exp = h('button', {}, 'Export table (CSV)');
    exp.addEventListener('click', () => this.exportCsv());
    const clr = h('button', {}, 'Clear');
    clr.addEventListener('click', () => { this.pinned = []; this.render(); });
    card.append(
      h('div', { class: 'modal-head' }, h('h2', {}, 'Studies — compare & sweep'), close),
      h('p', { class: 'dim' }, 'Pin results to compare configurations (A/B): the first pinned row is the baseline and the others show their difference. A sweep steps one parameter, lets the flow settle and averages the coefficients at every point.'),
      h('div', { class: 'btnrow ctl' }, pin, exp, clr),
      this.table,
      h('h3', {}, 'Parameter sweep'),
      this.sweepBox,
    );
    this.root.append(card);
    this.root.addEventListener('click', (e) => e.target === this.root && this.close());
    document.body.append(this.root);
    this.render();
    this.renderSweep();
  }

  open() {
    this.root.classList.remove('hidden');
    this.render();
  }
  close() {
    this.root.classList.add('hidden');
  }

  private cfg(): Cfg {
    const out: Record<string, unknown> = {};
    for (const k of CFG_KEYS) out[k] = this.app.s[k];
    return out as Cfg;
  }

  private describe(c: Cfg) {
    const veh = this.app.presetInfo?.name ?? (c.vehicle === 'upload' ? 'Uploaded' : c.vehicle);
    const bits = [`${veh}`, `${c.speedKmh.toFixed(0)} km/h`];
    if (c.yaw) bits.push(`yaw ${c.yaw.toFixed(1)}°`);
    if (c.pitch) bits.push(`pitch ${c.pitch.toFixed(1)}°`);
    if (c.rideMm) bits.push(`ride +${c.rideMm.toFixed(0)} mm`);
    bits.push(c.mode.toUpperCase());
    return bits.join(' · ');
  }

  async pinCurrent(label?: string) {
    const r = this.app.readout();
    if (!isFinite(r.cd)) {
      toast('No averaged result yet — let the flow settle first', 'warn');
      return;
    }
    const cfg = this.cfg();
    const thumb = await this.app.capture.thumbnail(200);
    this.pinned.push({
      label: label ?? this.describe(cfg), cfg, cd: r.cd, cdSE: r.cdSE, cl: r.cl, cs: r.cs, dragN: r.dragN,
      converged: r.converged, reSim: r.flow.reSim, thumb,
    });
    this.render();
    if (!label) toast(`Pinned: C_D ${r.cd.toFixed(3)}${r.converged ? '' : ' (not yet converged)'}`, 'info');
  }

  private async restore(p: Pinned) {
    const s = this.app.s;
    const needsRebuild = p.cfg.mode !== s.mode || p.cfg.quality2D !== s.quality2D || p.cfg.quality3D !== s.quality3D;
    const vehicleChanged = p.cfg.vehicle !== s.vehicle;
    Object.assign(s, p.cfg);
    if (needsRebuild) await this.app.rebuildSolver();
    else if (vehicleChanged) await this.app.setVehicle(p.cfg.vehicle);
    else {
      await this.app.revoxelize();
      this.app.flowChanged();
    }
    this.app.panel.sync();
    toast(`Restored: ${p.label}`, 'info');
  }

  private render() {
    const t = this.table;
    t.innerHTML = '';
    if (!this.pinned.length) {
      t.append(h('div', { class: 'dim small' }, 'No pinned results yet.'));
      return;
    }
    const base = this.pinned[0];
    const tbl = h('table', { class: 'metrics pins-t' });
    tbl.append(h('tr', {}, ...['', 'configuration', 'C_D', 'Δ C_D', 'C_L', 'C_S', 'drag', 'Re_sim', ''].map((x) => h('th', {}, x))));
    this.pinned.forEach((p, i) => {
      const img = h('img', { src: p.thumb, class: 'thumb' });
      const d = i === 0 ? 'baseline' : `${((p.cd / base.cd - 1) * 100).toFixed(1)} %`;
      const rest = h('button', { title: 'Apply this configuration' }, '↺');
      rest.addEventListener('click', () => this.restore(p));
      const del = h('button', { title: 'Remove' }, '✕');
      del.addEventListener('click', () => { this.pinned.splice(i, 1); this.render(); });
      tbl.append(h('tr', {},
        h('td', {}, img),
        h('td', {}, p.label, p.converged ? '' : h('span', { class: 'warnv' }, ' (unconverged)')),
        h('td', { class: 'mono' }, `${p.cd.toFixed(3)}${isFinite(p.cdSE) ? ' ± ' + p.cdSE.toFixed(3) : ''}`),
        h('td', { class: 'mono' }, d),
        h('td', { class: 'mono' }, p.cl.toFixed(3)),
        h('td', { class: 'mono' }, isFinite(p.cs) ? p.cs.toFixed(3) : '—'),
        h('td', { class: 'mono' }, isFinite(p.dragN) ? `${p.dragN.toFixed(0)} N` : '—'),
        h('td', { class: 'mono' }, p.reSim.toFixed(0)),
        h('td', {}, rest, del),
      ));
    });
    t.append(tbl);
  }

  private exportCsv() {
    const rows = this.pinned.map((p) => [p.label, p.cd, p.cdSE, p.cl, p.cs, p.dragN, p.reSim, String(p.converged), ...CFG_KEYS.map((k) => String(p.cfg[k]))]);
    download(csv(['label', 'cd', 'cd_se', 'cl', 'cs', 'drag_N', 're_sim', 'converged', ...CFG_KEYS], rows), `windtunnel-results-${stamp()}.csv`);
  }

  private renderSweep() {
    const box = this.sweepBox;
    box.innerHTML = '';
    const sel = h('select');
    for (const [k, v] of Object.entries(PARAMS)) sel.append(h('option', { value: k }, v.label));
    const num = (v: number, step = 'any') => { const i = h('input', { type: 'number', step, class: 'num' }); i.value = String(v); return i; };
    const p0 = PARAMS.yaw;
    const from = num(p0.from), to = num(p0.to), n = num(p0.n, '1'), settle = num(2), avg = num(3);
    sel.addEventListener('change', () => {
      const p = PARAMS[sel.value as SweepParam];
      from.value = String(p.from); to.value = String(p.to); n.value = String(p.n);
    });
    const run = h('button', { class: 'accent' }, this.running ? 'Abort' : 'Run sweep');
    run.addEventListener('click', () => {
      if (this.running) { this.abortFlag = true; return; }
      const param = sel.value as SweepParam;
      const N = Math.max(2, Math.min(20, Math.round(+n.value)));
      const a = +from.value, b = +to.value;
      const values = Array.from({ length: N }, (_, i) => a + ((b - a) * i) / (N - 1));
      this.runSweep(param, values, Math.max(0.5, +settle.value), Math.max(1, +avg.value));
    });
    const lab = (t: string, e: HTMLElement) => h('label', { class: 'sw-f' }, h('span', { class: 'dim small' }, t), e);
    box.append(
      h('div', { class: 'sw-row' }, lab('parameter', sel), lab('from', from), lab('to', to), lab('points', n), lab('settle t·U/L', settle), lab('average t·U/L', avg), run),
      h('div', { class: 'dim small', id: 'sweep-status' }, this.running ? 'running…' : 'Each point: apply → settle → average. On a fast GPU a 4-point 3D sweep takes a few minutes.'),
    );
    if (this.sweep?.pts.length) {
      const P = PARAMS[this.sweep.param];
      const cbox = h('div', { class: 'chartbox' }, h('div', { class: 'chart-t' }, h('span', {}, `C_D (amber) and C_L (cyan) vs ${P.label} [${P.unit}]`), h('span', { class: 'dim' }, `${this.sweep.pts.length} points`)));
      box.append(cbox);
      const ch = new LineChart(cbox, 140);
      const t = this.sweep.pts.map((q) => q.v);
      requestAnimationFrame(() => ch.draw([
        { t, v: this.sweep!.pts.map((q) => q.cd), color: '#ffd166', width: 2 },
        { t, v: this.sweep!.pts.map((q) => q.cl), color: '#5ad1ff', width: 1.5 },
      ]));
      const tbl = h('table', { class: 'metrics' });
      tbl.append(h('tr', {}, ...[P.label, 'C_D', 'C_L', 'C_S'].map((x) => h('th', {}, x))));
      for (const q of this.sweep.pts) tbl.append(h('tr', {}, h('td', {}, `${q.v.toFixed(2)} ${P.unit}`), h('td', { class: 'mono' }, `${q.cd.toFixed(3)} ± ${q.cdSE.toFixed(3)}`), h('td', { class: 'mono' }, q.cl.toFixed(3)), h('td', { class: 'mono' }, isFinite(q.cs) ? q.cs.toFixed(3) : '—')));
      box.append(tbl);
    }
  }

  private status(t: string) {
    const e = document.getElementById('sweep-status');
    if (e) e.textContent = t;
  }

  private async apply(param: SweepParam, v: number) {
    const s = this.app.s;
    if (param === 'speedKmh') {
      s.speedKmh = v;
      this.app.flowChanged();
    } else {
      if (param === 'yaw') s.yaw = v;
      if (param === 'pitch') s.pitch = v;
      if (param === 'rideMm') s.rideMm = Math.max(0, v);
      await this.app.revoxelize();
    }
    this.app.panel.sync();
  }

  private async runSweep(param: SweepParam, values: number[], settle: number, avg: number) {
    if (param === 'yaw' && !this.app.is3D) {
      toast('Yaw sweeps need the 3D solver', 'warn');
      return;
    }
    this.running = true;
    this.abortFlag = false;
    this.sweep = { param, pts: [] };
    this.app.s.paused = false;
    this.renderSweep();
    const P = PARAMS[param];
    for (let i = 0; i < values.length && !this.abortFlag; i++) {
      const v = values[i];
      await this.apply(param, v);
      const t0 = this.app.clock + settle;
      while (!this.abortFlag && this.app.clock < t0 + avg) {
        const left = Math.max(0, t0 + avg - this.app.clock);
        this.status(`point ${i + 1}/${values.length}: ${P.label} = ${v.toFixed(2)} ${P.unit} — ${this.app.clock < t0 ? 'settling' : 'averaging'}, ${left.toFixed(1)} t·U/L left`);
        await new Promise((r) => setTimeout(r, 300));
      }
      if (this.abortFlag) break;
      const st = this.app.statsSince(t0);
      this.sweep.pts.push({ v, cd: st.cd.mean, cdSE: st.cd.se, cl: st.cl.mean, cs: st.cs.mean });
      await this.pinCurrent(`${this.describe(this.cfg())} (sweep)`);
      this.renderSweep();
    }
    this.running = false;
    this.renderSweep();
    this.status(this.abortFlag ? 'aborted' : 'sweep finished — every point was also pinned to the results table');
  }
}
