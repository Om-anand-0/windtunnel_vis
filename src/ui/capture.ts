import type { App } from '../app';
import { colormapCSS, ColormapName } from '../render/colormaps';
import { FIELDS } from '../state';
import { download, stamp } from '../analysis/export';
import { toast } from './widgets';

/**
 * Screenshot (PNG) and video (WebM via MediaRecorder) capture. Frames are composited from the
 * WebGPU canvas right after submit (same task, so the swap-chain texture is still readable) plus a
 * burned-in legend and the aero numbers.
 */
export class Capture {
  recording = false;
  private shotPending = false;
  private rec: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private recCanvas: HTMLCanvasElement | null = null;

  constructor(private app: App) {}

  screenshot() {
    this.shotPending = true;
  }

  private thumbWaiters: { w: number; res: (url: string) => void }[] = [];

  /** Small JPEG of the next frame (for pinned results). */
  thumbnail(w = 240): Promise<string> {
    return new Promise((res) => this.thumbWaiters.push({ w, res }));
  }

  toggleRecording() {
    if (this.recording) {
      this.rec?.stop();
      this.recording = false;
      return;
    }
    const src = this.app.backend.canvas;
    const c = document.createElement('canvas');
    c.width = src.width & ~1;
    c.height = src.height & ~1;
    this.recCanvas = c;
    const stream = c.captureStream(60);
    const types = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
    const mime = types.find((t) => MediaRecorder.isTypeSupported(t));
    if (!mime) {
      toast('Video recording is not supported in this browser', 'error');
      return;
    }
    this.chunks = [];
    this.rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 16_000_000 });
    this.rec.ondataavailable = (e) => e.data.size && this.chunks.push(e.data);
    this.rec.onstop = () => {
      const blob = new Blob(this.chunks, { type: mime });
      download(blob, `windtunnel-${stamp()}.${mime.includes('mp4') ? 'mp4' : 'webm'}`);
      this.recCanvas = null;
    };
    this.rec.start(250);
    this.recording = true;
    toast('Recording… press “Stop recording” to save', 'info');
  }

  private gif: { frames: ImageData[]; canvas: HTMLCanvasElement; next: number; done: (on: boolean) => void } | null = null;

  /** ~6 s animated GIF at 12.5 fps, max 720 px wide, encoded with gifenc (per-frame palettes). */
  recordGif(state: (on: boolean) => void) {
    if (this.gif) return;
    const src = this.app.backend.canvas;
    const scale = Math.min(1, 720 / src.width);
    const c = document.createElement('canvas');
    c.width = Math.round(src.width * scale) & ~1;
    c.height = Math.round(src.height * scale) & ~1;
    this.gif = { frames: [], canvas: c, next: performance.now(), done: state };
    state(true);
  }

  private async encodeGif(frames: ImageData[], done: (on: boolean) => void) {
    toast(`Encoding GIF (${frames.length} frames)…`, 'info', 3000);
    const { GIFEncoder, quantize, applyPalette } = await import('gifenc');
    const enc = GIFEncoder();
    for (const f of frames) {
      const palette = quantize(f.data, 256);
      enc.writeFrame(applyPalette(f.data, palette), f.width, f.height, { palette, delay: 80 });
      await new Promise((r) => setTimeout(r, 0));
    }
    enc.finish();
    download(new Blob([enc.bytes() as BlobPart], { type: 'image/gif' }), `windtunnel-${stamp()}.gif`);
    done(false);
  }

  afterFrame() {
    if (this.thumbWaiters.length) {
      const src = this.app.backend.canvas;
      for (const t of this.thumbWaiters.splice(0)) {
        const c = document.createElement('canvas');
        c.width = t.w;
        c.height = Math.round((t.w * src.height) / Math.max(src.width, 1));
        c.getContext('2d')!.drawImage(src, 0, 0, c.width, c.height);
        t.res(c.toDataURL('image/jpeg', 0.8));
      }
    }
    if (this.gif && performance.now() >= this.gif.next) {
      const g = this.gif;
      g.next += 80;
      if (performance.now() - g.next > 200) g.next = performance.now() + 80;
      this.composite(g.canvas);
      g.frames.push(g.canvas.getContext('2d')!.getImageData(0, 0, g.canvas.width, g.canvas.height));
      if (g.frames.length >= 75) {
        this.gif = null;
        this.encodeGif(g.frames, g.done);
      }
    }
    if (this.shotPending) {
      this.shotPending = false;
      const c = document.createElement('canvas');
      c.width = this.app.backend.canvas.width;
      c.height = this.app.backend.canvas.height;
      this.composite(c);
      c.toBlob((b) => b && download(b, `windtunnel-${stamp()}.png`), 'image/png');
    }
    if (this.recording && this.recCanvas) this.composite(this.recCanvas);
  }

  private composite(c: HTMLCanvasElement) {
    const g = c.getContext('2d')!;
    const a = this.app;
    g.drawImage(a.backend.canvas, 0, 0, c.width, c.height);
    const k = c.width / Math.max(a.backend.canvas.clientWidth, 1);
    const r = a.readout();
    const s = a.s;
    // header
    g.fillStyle = 'rgba(10,12,16,0.72)';
    g.fillRect(12 * k, 12 * k, 330 * k, 64 * k);
    g.fillStyle = '#e6edf6';
    g.font = `${600} ${14 * k}px Inter, system-ui, sans-serif`;
    const veh = a.presetInfo?.name ?? 'Uploaded model';
    g.fillText(`${veh} · ${s.speedKmh.toFixed(0)} km/h · ${s.mode.toUpperCase()}`, 22 * k, 32 * k);
    g.font = `${12 * k}px ui-monospace, Menlo, monospace`;
    g.fillStyle = '#ffd166';
    const f = (v: number) => (isFinite(v) ? v.toFixed(3) : '—');
    g.fillText(`Cd ${f(r.cd)}   Cl ${f(r.cl)}   Re_sim ${Math.round(r.flow.reSim).toLocaleString()}`, 22 * k, 52 * k);
    g.fillStyle = '#9fb0c8';
    g.fillText(`drag ${isFinite(r.dragN) ? r.dragN.toFixed(0) : '—'} N · A ${r.frontalM2.toFixed(2)} m² · t·U/L ${r.convTime.toFixed(1)}`, 22 * k, 68 * k);
    // legend
    if (s.sliceOn) {
      const fd = FIELDS[s.field];
      const w = 260 * k, hgt = 10 * k, x = c.width / 2 - w / 2, y = c.height - 40 * k;
      g.fillStyle = 'rgba(10,12,16,0.72)';
      g.fillRect(x - 10 * k, y - 22 * k, w + 20 * k, 50 * k);
      const grad = g.createLinearGradient(x, 0, x + w, 0);
      const css = colormapCSS(s.cmap as ColormapName, 12);
      const stops = css.slice(css.indexOf('(') + 1, -1).split(/,(?![^(]*\))/).slice(1);
      for (const st of stops) {
        const m = st.trim().match(/(rgb\([^)]*\))\s+([\d.]+)%/);
        if (m) grad.addColorStop(parseFloat(m[2]) / 100, m[1]);
      }
      g.fillStyle = grad;
      g.fillRect(x, y, w, hgt);
      g.fillStyle = '#e6edf6';
      g.font = `${11 * k}px Inter, system-ui, sans-serif`;
      g.fillText(`${fd.label} (${fd.unit})`, x, y - 6 * k);
      g.fillText(s.vmin.toFixed(2), x, y + hgt + 13 * k);
      const t = s.vmax.toFixed(2);
      g.fillText(t, x + w - g.measureText(t).width, y + hgt + 13 * k);
    }
  }
}
