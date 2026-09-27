/**
 * Headless physics + voxelizer checks (used by CI). Runs the solver on SwiftShader's software WebGPU,
 * so grids are kept small. Usage:
 *   npm test                  quick: free stream, voxelizer volumes, small cylinder (St)
 *   npm test -- --full        adds the sphere drag case
 *   npm test -- --case=car&vehicle=f1&dims=256,96,1&steps=2000   any harness query string
 */
import { createServer } from 'vite';
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const custom = args.find((a) => a.startsWith('--case='));
const query = custom
  ? custom.slice(2)
  : `case=freestream,voxcheck,cylinder${args.includes('--full') ? ',sphere' : ''}&valscale=0.5`;

// no HMR / file watching: editing sources while a long run is in progress must not restart it
const server = await createServer({ server: { port: 0, hmr: false, watch: null }, logLevel: 'error' });
await server.listen();
const port = server.httpServer.address().port;
const executablePath = process.env.CHROMIUM_PATH || undefined;
const browser = await chromium.launch({
  headless: true,
  channel: executablePath ? undefined : 'chromium',
  executablePath,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader', '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage();
page.on('console', (m) => {
  const t = m.text();
  if (m.type() === 'error' && !t.includes('404')) console.log('[browser]', t);
  else if (t.startsWith('[progress]')) console.log(t);
});
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
const t0 = Date.now();
let code = 0;
try {
  await page.goto(`http://localhost:${port}/test.html?${query}`);
  await page.waitForFunction(() => window.__done === true, null, { timeout: 60 * 60 * 1000 });
  const out = await page.evaluate(() => document.getElementById('out').textContent);
  console.log(out);
  const results = await page.evaluate(() => window.__results ?? []);
  const failed = results.filter((r) => r.passed === false);
  if (!results.length) { console.log('no results'); code = 1; }
  for (const r of results) console.log(`${r.passed === false ? 'FAIL' : 'ok  '} ${r.name ?? r.id}`);
  if (failed.length) code = 1;
} catch (e) {
  console.error(e);
  code = 1;
}
console.log(`finished in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
await browser.close();
await server.close();
process.exit(code);
