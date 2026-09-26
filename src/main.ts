import './style.css';
import { initWebGPU } from './gpu/device';

async function boot() {
  const msg = document.getElementById('boot-msg')!;
  const canvas = document.getElementById('gpu') as HTMLCanvasElement;
  try {
    const gpu = await initWebGPU();
    if (gpu) {
      const { App } = await import('./app');
      const { WebGPUBackend } = await import('./backend/webgpu');
      const app = new App(new WebGPUBackend(gpu, canvas));
      (window as unknown as { app: unknown }).app = app;
      await app.init();
      gpu.lost.then((info) => {
        msg.textContent = `GPU device lost: ${info.message}. Reload the page.`;
        document.getElementById('boot')!.style.display = '';
      });
    } else {
      msg.textContent = 'WebGPU unavailable — starting WebGL2 fallback (2D)…';
      const { App } = await import('./app');
      const { WebGLBackend } = await import('./backend/webgl');
      const app = new App(new WebGLBackend(canvas));
      (window as unknown as { app: unknown }).app = app;
      await app.init();
    }
    document.getElementById('boot')!.style.display = 'none';
  } catch (e) {
    console.error(e);
    msg.textContent = `Failed to start: ${(e as Error).message ?? e}`;
  }
}
boot();
