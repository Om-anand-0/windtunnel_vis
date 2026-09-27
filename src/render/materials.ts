import type { MaterialDef, TextureImage } from '../voxelize/mesh';

/** Look of the procedural presets in "Materials" mode: a glossy metallic paint. */
export const PRESET_PAINT: MaterialDef = {
  color: [0.06, 0.14, 0.3, 1],
  flipY: false,
  emissive: [0, 0, 0],
  metalness: 0.55,
  roughness: 0.28,
  alphaMode: 0,
  alphaCutoff: 0.5,
  vertexColors: false,
};

/** Textures are capped at this size: a detailed car can carry dozens of 4K maps. */
export const MAX_TEXTURE = 2048;

/** 64-byte uniform block matching `struct Mat` in MESH_WGSL. */
export function materialUniform(m: MaterialDef, hasMap: boolean): ArrayBuffer {
  const b = new ArrayBuffer(64);
  const f = new Float32Array(b), u = new Uint32Array(b);
  f.set(m.color, 0);
  f.set(m.emissive, 4);
  f[8] = m.metalness;
  f[9] = m.roughness;
  u[10] = m.alphaMode;
  f[11] = m.alphaCutoff;
  u[12] = hasMap ? 1 : 0;
  u[13] = m.vertexColors ? 1 : 0;
  return b;
}

const dims = (img: TextureImage): [number, number] => {
  const el = img as HTMLImageElement;
  return [el.naturalWidth || img.width, el.naturalHeight || img.height];
};

/**
 * Upload a colour map as an sRGB texture with a full mip chain. Each level is resampled from the
 * source by the browser (createImageBitmap), so no mip-generation shader is needed.
 */
export async function textureFromImage(device: GPUDevice, img: TextureImage, flipY: boolean): Promise<GPUTexture> {
  const [w0, h0] = dims(img);
  const scale = Math.min(1, MAX_TEXTURE / Math.max(w0, h0, 1), device.limits.maxTextureDimension2D / Math.max(w0, h0, 1));
  const w = Math.max(1, Math.round(w0 * scale)), h = Math.max(1, Math.round(h0 * scale));
  const levels = Math.floor(Math.log2(Math.max(w, h))) + 1;
  const tex = device.createTexture({
    label: 'material map',
    size: [w, h],
    format: 'rgba8unorm-srgb',
    mipLevelCount: levels,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  for (let l = 0; l < levels; l++) {
    const lw = Math.max(1, w >> l), lh = Math.max(1, h >> l);
    const bmp = await createImageBitmap(img, { resizeWidth: lw, resizeHeight: lh, resizeQuality: 'high', premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
    device.queue.copyExternalImageToTexture({ source: bmp, flipY }, { texture: tex, mipLevel: l }, [lw, lh]);
    bmp.close();
  }
  return tex;
}
