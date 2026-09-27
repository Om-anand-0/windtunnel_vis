export interface GpuContext {
  device: GPUDevice;
  adapter: GPUAdapter;
  adapterInfo: string;
  maxBinding: number;
  maxBuffer: number;
  hasTimestamps: boolean;
  lost: Promise<GPUDeviceLostInfo>;
}

/** Try to create a WebGPU device with the largest storage limits the adapter allows. */
export async function initWebGPU(): Promise<GpuContext | null> {
  if (!('gpu' in navigator) || !navigator.gpu) return null;
  const params = new URLSearchParams(location.search);
  if (params.has('webgl')) return null;
  let adapter: GPUAdapter | null = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch {
    adapter = null;
  }
  if (!adapter) return null;
  const want: Record<string, number> = {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
    maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
    maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 10),
    maxComputeInvocationsPerWorkgroup: Math.min(adapter.limits.maxComputeInvocationsPerWorkgroup, 256),
    maxComputeWorkgroupSizeX: Math.min(adapter.limits.maxComputeWorkgroupSizeX, 256),
  };
  const features: GPUFeatureName[] = [];
  const hasTimestamps = adapter.features.has('timestamp-query');
  if (hasTimestamps) features.push('timestamp-query');
  let device: GPUDevice;
  try {
    device = await adapter.requestDevice({ requiredLimits: want, requiredFeatures: features });
  } catch (e) {
    console.warn('requestDevice with raised limits failed, retrying with defaults', e);
    device = await adapter.requestDevice();
  }
  const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
  const adapterInfo = info ? [info.vendor, info.architecture, info.description].filter(Boolean).join(' ') : 'WebGPU';
  device.addEventListener('uncapturederror', (ev) => {
    const msg = (ev as GPUUncapturedErrorEvent).error.message;
    console.error('WebGPU error:', msg);
    window.dispatchEvent(new CustomEvent('gpu-error', { detail: msg }));
  });
  // keep a global reference: some Chromium builds drop the instance if the adapter is collected
  (globalThis as unknown as { __gpuAdapter: GPUAdapter }).__gpuAdapter = adapter;
  return {
    device,
    adapter,
    adapterInfo: adapterInfo || 'WebGPU adapter',
    maxBinding: device.limits.maxStorageBufferBindingSize,
    maxBuffer: device.limits.maxBufferSize,
    hasTimestamps: device.features.has('timestamp-query'),
    lost: device.lost,
  };
}

/** Split a 1D workgroup count into a 2D dispatch that respects the 65535 limit. */
export function dispatch1D(cells: number, wg: number): { x: number; y: number; strideX: number } {
  const groups = Math.ceil(cells / wg);
  const x = Math.min(groups, 65535);
  const y = Math.ceil(groups / x);
  return { x, y, strideX: x * wg };
}
