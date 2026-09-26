/** Tiny DOM widget kit — no framework, just typed helpers that keep the panel code readable. */

export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, string> = {}, ...children: (Node | string)[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') el.className = v;
    else el.setAttribute(k, v);
  }
  for (const c of children) el.append(c);
  return el;
}

export interface Control<T> {
  el: HTMLElement;
  set(v: T): void;
  get(): T;
  setEnabled(on: boolean): void;
  setVisible(on: boolean): void;
}

export function section(parent: HTMLElement, title: string, open = true): HTMLElement {
  const det = h('details', { class: 'sec' });
  det.open = open;
  const sum = h('summary', {}, title);
  const body = h('div', { class: 'sec-body' });
  det.append(sum, body);
  parent.append(det);
  return body;
}

function common<T>(el: HTMLElement, get: () => T, set: (v: T) => void, inputs: (HTMLInputElement | HTMLSelectElement | HTMLButtonElement)[]): Control<T> {
  return {
    el, get, set,
    setEnabled(on) {
      el.classList.toggle('disabled', !on);
      for (const i of inputs) i.disabled = !on;
    },
    setVisible(on) {
      el.style.display = on ? '' : 'none';
    },
  };
}

export function slider(parent: HTMLElement, o: {
  label: string; min: number; max: number; step: number; value: number;
  format?: (v: number) => string; onInput: (v: number) => void; hint?: string; log?: boolean;
}): Control<number> & { setHint(s: string): void } {
  const toPos = (v: number) => (o.log ? Math.log10(v) : v);
  const fromPos = (p: number) => (o.log ? Math.pow(10, p) : p);
  const input = h('input', { type: 'range', min: String(toPos(o.min)), max: String(toPos(o.max)), step: String(o.log ? (toPos(o.max) - toPos(o.min)) / 200 : o.step) });
  input.value = String(toPos(o.value));
  const fmt = o.format ?? ((v: number) => String(v));
  const val = h('span', { class: 'val' }, fmt(o.value));
  const hint = h('div', { class: 'hint' }, o.hint ?? '');
  const row = h('div', { class: 'ctl slider' }, h('div', { class: 'lab' }, h('span', {}, o.label), val), input, hint);
  if (!o.hint) hint.style.display = 'none';
  let cur = o.value;
  const fill = () => {
    const p = (parseFloat(input.value) - parseFloat(input.min)) / (parseFloat(input.max) - parseFloat(input.min));
    input.style.setProperty('--p', `${(p * 100).toFixed(1)}%`);
  };
  fill();
  input.addEventListener('input', () => {
    let v = fromPos(parseFloat(input.value));
    if (!o.log) v = Math.round(v / o.step) * o.step;
    cur = v;
    val.textContent = fmt(v);
    fill();
    o.onInput(v);
  });
  parent.append(row);
  const c = common(row, () => cur, (v) => {
    cur = v;
    input.value = String(toPos(v));
    val.textContent = fmt(v);
    fill();
  }, [input]);
  return Object.assign(c, {
    setHint(s: string) {
      hint.textContent = s;
      hint.style.display = s ? '' : 'none';
    },
  });
}

export function select<T extends string | number>(parent: HTMLElement, o: {
  label: string; options: { value: T; label: string }[]; value: T; onChange: (v: T) => void;
}): Control<T> & { setOptions(opts: { value: T; label: string }[]): void } {
  const sel = h('select');
  const fill = (opts: { value: T; label: string }[]) => {
    sel.innerHTML = '';
    for (const op of opts) {
      const e = h('option', { value: String(op.value) }, op.label);
      sel.append(e);
    }
  };
  fill(o.options);
  let opts = o.options;
  sel.value = String(o.value);
  sel.addEventListener('change', () => {
    const op = opts.find((x) => String(x.value) === sel.value);
    if (op) o.onChange(op.value);
  });
  const row = h('label', { class: 'ctl select' }, h('span', { class: 'lab1' }, o.label), sel);
  parent.append(row);
  const c = common<T>(row, () => opts.find((x) => String(x.value) === sel.value)!.value, (v) => (sel.value = String(v)), [sel]);
  return Object.assign(c, {
    setOptions(n: { value: T; label: string }[]) {
      const v = sel.value;
      opts = n;
      fill(n);
      sel.value = v;
    },
  });
}

export function toggle(parent: HTMLElement, o: { label: string; value: boolean; onChange: (v: boolean) => void; title?: string }): Control<boolean> {
  const input = h('input', { type: 'checkbox' });
  input.checked = o.value;
  input.addEventListener('change', () => o.onChange(input.checked));
  const row = h('label', { class: 'ctl toggle', title: o.title ?? '' }, input, h('span', { class: 'sw' }), h('span', { class: 'lab1' }, o.label));
  parent.append(row);
  return common(row, () => input.checked, (v) => (input.checked = v), [input]);
}

export function segmented<T extends string | number>(parent: HTMLElement, o: {
  label?: string; options: { value: T; label: string; title?: string }[]; value: T; onChange: (v: T) => void;
}): Control<T> {
  const wrap = h('div', { class: 'seg' });
  let cur = o.value;
  const btns = o.options.map((op) => {
    const b = h('button', { type: 'button', title: op.title ?? op.label }, op.label);
    b.addEventListener('click', () => {
      cur = op.value;
      refresh();
      o.onChange(op.value);
    });
    wrap.append(b);
    return b;
  });
  const refresh = () => btns.forEach((b, i) => b.classList.toggle('on', o.options[i].value === cur));
  refresh();
  const row = h('div', { class: 'ctl segrow' });
  if (o.label) row.append(h('span', { class: 'lab1' }, o.label));
  row.append(wrap);
  parent.append(row);
  return common(row, () => cur, (v) => { cur = v; refresh(); }, btns);
}

export function buttonRow(parent: HTMLElement, items: { label: string; onClick: () => void; title?: string; cls?: string }[]): HTMLButtonElement[] {
  const row = h('div', { class: 'ctl btnrow' });
  const out = items.map((it) => {
    const b = h('button', { type: 'button', title: it.title ?? '', class: it.cls ?? '' }, it.label);
    b.addEventListener('click', it.onClick);
    row.append(b);
    return b;
  });
  parent.append(row);
  return out;
}

export function note(parent: HTMLElement, text: string): HTMLElement {
  const e = h('div', { class: 'note' }, text);
  parent.append(e);
  return e;
}

export function toast(msg: string, kind: 'info' | 'warn' | 'error' = 'info', ms = 4500) {
  let host = document.getElementById('toasts');
  if (!host) {
    host = h('div', { id: 'toasts' });
    document.body.append(host);
  }
  const t = h('div', { class: `toast ${kind}` }, msg);
  host.append(t);
  requestAnimationFrame(() => t.classList.add('show'));
  setTimeout(() => {
    t.classList.remove('show');
    setTimeout(() => t.remove(), 400);
  }, ms);
}
