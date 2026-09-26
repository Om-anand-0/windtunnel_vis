import type { App } from '../app';
import { CASES, ValidationResult } from '../analysis/validation';
import { LineChart } from './chart';
import { h } from './widgets';

/** Modal that runs the physics validation cases on dedicated solver instances and reports pass/fail. */
export class ValidationPanel {
  private root: HTMLElement;
  private list: HTMLElement;
  private running = false;
  private abortFlag = false;
  private runBtn: HTMLButtonElement;
  results = new Map<string, ValidationResult>();

  constructor(private app: App) {
    this.root = h('div', { class: 'modal hidden' });
    const card = h('div', { class: 'modal-card' });
    const close = h('button', { class: 'close', title: 'Close' }, '✕');
    close.addEventListener('click', () => this.close());
    this.runBtn = h('button', { class: 'accent' }, 'Run all checks');
    this.runBtn.addEventListener('click', () => (this.running ? (this.abortFlag = true) : this.runAll()));
    this.list = h('div', { class: 'val-list' });
    card.append(
      h('div', { class: 'modal-head' }, h('h2', {}, 'Validation'), close),
      h('p', { class: 'dim' },
        'Each check runs on its own solver instance with analytic geometry, independent of the scene. ' +
        'The main simulation is paused while checks run.'),
      this.runBtn,
      this.list,
    );
    this.root.append(card);
    this.root.addEventListener('click', (e) => e.target === this.root && this.close());
    document.body.append(this.root);
    this.renderList();
  }

  open() {
    this.root.classList.remove('hidden');
  }

  close() {
    this.root.classList.add('hidden');
  }

  private renderList(progress?: { id: string; f: number; msg?: string }) {
    this.list.innerHTML = '';
    for (const c of CASES) {
      const r = this.results.get(c.id);
      const badge = r ? h('span', { class: `badge ${r.passed ? 'pass' : 'fail'}` }, r.passed ? 'PASS' : 'FAIL') : h('span', { class: 'badge idle' }, progress?.id === c.id ? 'RUNNING' : '—');
      const item = h('div', { class: 'val-item' }, h('div', { class: 'val-h' }, badge, h('b', {}, c.name)), h('div', { class: 'dim small' }, c.description));
      if (progress?.id === c.id) {
        const bar = h('div', { class: 'pbar' }, h('div', { style: `width:${(progress.f * 100).toFixed(1)}%` }));
        item.append(bar, h('div', { class: 'dim small' }, progress.msg ?? ''));
      }
      if (r) {
        const tbl = h('table', { class: 'metrics' });
        for (const m of r.metrics) tbl.append(h('tr', {}, h('td', {}, m.label), h('td', { class: 'mono' }, m.value), h('td', { class: 'dim' }, m.expected ?? '')));
        tbl.append(h('tr', {}, h('td', {}, 'run time'), h('td', { class: 'mono' }, `${r.seconds.toFixed(1)} s`), h('td', {}, '')));
        item.append(tbl);
        if (r.series && r.series.t.length > 4) {
          const box = h('div', { class: 'chartbox' }, h('div', { class: 'chart-t' }, h('span', {}, r.series.label), h('span', { class: 'dim' }, 'vs. step')));
          item.append(box);
          requestAnimationFrame(() => new LineChart(box, 80).draw([{ t: r.series!.t, v: r.series!.v, color: '#5ad1ff', width: 1.4 }]));
        }
        if (r.error) item.append(h('div', { class: 'err' }, r.error));
      }
      this.list.append(item);
    }
  }

  async runAll() {
    this.running = true;
    this.abortFlag = false;
    this.runBtn.textContent = 'Abort';
    const wasPaused = this.app.s.paused;
    this.app.s.paused = true;
    this.app.panel.sync();
    for (const c of CASES) {
      if (this.abortFlag) break;
      this.renderList({ id: c.id, f: 0 });
      let lastDraw = 0;
      try {
        if (c.needs3D && !this.app.backend.supports3D) {
          this.results.set(c.id, { id: c.id, name: c.name, passed: false, metrics: [], error: 'Skipped: needs the WebGPU 3D solver', seconds: 0 });
          this.renderList();
          continue;
        }
        const r = await c.run(this.app.backend.solverFactory(), (f, msg) => {
          const now = performance.now();
          if (now - lastDraw > 150) {
            lastDraw = now;
            this.renderList({ id: c.id, f, msg });
          }
        }, () => this.abortFlag);
        this.results.set(c.id, r);
      } catch (e) {
        this.results.set(c.id, { id: c.id, name: c.name, passed: false, metrics: [], error: String(e), seconds: 0 });
      }
      this.renderList();
    }
    this.app.s.paused = wasPaused;
    this.app.panel.sync();
    this.running = false;
    this.runBtn.textContent = 'Run all checks';
    this.app.hud.update();
  }
}
