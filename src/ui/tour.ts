import { h } from './widgets';

/** Short first-run walkthrough. Re-open it with the “?” button in the top bar. */
const KEY = 'windtunnel.tour.done';

interface Step {
  target: () => Element | null;
  title: string;
  text: string;
}

const STEPS: Step[] = [
  {
    target: () => document.getElementById('view'),
    title: 'A real wind tunnel on your GPU',
    text: 'This is a lattice Boltzmann CFD simulation, not an animation. Drag to orbit, scroll to zoom, right-drag to pan. Drag the yellow handle to move the smoke rake.',
  },
  {
    target: () => document.getElementById('panel'),
    title: 'Controls',
    text: 'Vehicle, wind speed, heatmap field, smoke, streamlines and vortex surfaces all live here. Upload your own .glb/.obj/.stl, or drop a file on the page. Hover any control to see what it does.',
  },
  {
    target: () => document.getElementById('hud'),
    title: 'Aerodynamic results',
    text: 'Drag, lift and side-force coefficients come from momentum exchange on the body, with a statistical error bar. Wait for the green “converged” badge before trusting C_D.',
  },
  {
    target: () => document.getElementById('topbar'),
    title: 'Performance',
    text: 'Frame rate, solver steps per frame and MLUPS (million lattice updates per second). The number of steps adapts to hold the target frame rate.',
  },
  {
    target: () => Array.from(document.querySelectorAll('#panel button')).find((b) => b.textContent?.startsWith('Validation')) ?? null,
    title: 'Proof it’s real physics',
    text: 'The validation suite reproduces textbook results: the von Kármán vortex street behind a cylinder, the drag of a sphere, and the Ahmed reference car body.',
  },
];

export function maybeStartTour() {
  try {
    if (localStorage.getItem(KEY)) return;
  } catch {
    return;
  }
  startTour();
}

export function startTour() {
  document.getElementById('tour')?.remove();
  const root = h('div', { id: 'tour' });
  const spot = h('div', { class: 'tour-spot' });
  const card = h('div', { class: 'tour-card' });
  root.append(spot, card);
  document.body.append(root);
  let i = 0;
  const finish = () => {
    root.remove();
    try {
      localStorage.setItem(KEY, '1');
    } catch {
      /* ignore */
    }
  };
  const show = () => {
    const st = STEPS[i];
    const el = st.target();
    if (el instanceof HTMLElement) el.closest('details')?.setAttribute('open', '');
    if (el && 'scrollIntoView' in el) (el as HTMLElement).scrollIntoView({ block: 'nearest' });
    const r = el?.getBoundingClientRect() ?? new DOMRect(innerWidth / 2 - 100, innerHeight / 2 - 50, 200, 100);
    const pad = 6;
    Object.assign(spot.style, { left: `${r.left - pad}px`, top: `${r.top - pad}px`, width: `${r.width + 2 * pad}px`, height: `${r.height + 2 * pad}px` });
    card.innerHTML = '';
    const next = h('button', { class: 'accent' }, i === STEPS.length - 1 ? 'Done' : 'Next');
    const skip = h('button', {}, 'Skip');
    next.addEventListener('click', () => (++i >= STEPS.length ? finish() : show()));
    skip.addEventListener('click', finish);
    card.append(
      h('div', { class: 'tour-step' }, `${i + 1} / ${STEPS.length}`),
      h('h3', {}, st.title),
      h('p', {}, st.text),
      h('div', { class: 'tour-btns' }, skip, next),
    );
    // place the card beside the highlighted area, inside the viewport
    const cw = 320, ch = card.offsetHeight || 170;
    let x = r.right + 16, y = r.top;
    if (x + cw > innerWidth - 8) x = r.left - cw - 16;
    if (x < 8) { x = Math.min(Math.max(r.left, 8), innerWidth - cw - 8); y = r.bottom + 12; }
    if (r.height > innerHeight * 0.6) y = r.top + r.height / 2 - ch / 2;
    card.style.left = `${Math.max(8, x)}px`;
    card.style.top = `${Math.min(Math.max(8, y), innerHeight - ch - 8)}px`;
  };
  window.addEventListener('keydown', function esc(e) {
    if (e.key === 'Escape') { finish(); window.removeEventListener('keydown', esc); }
  });
  show();
}
