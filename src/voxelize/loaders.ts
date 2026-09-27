import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACO_GLTF_CONFIG, DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';
import { mergeObject, RawMesh } from './mesh';

export interface LoadedModel {
  name: string;
  raw: RawMesh;
  /** best guess for the up axis based on file conventions */
  up: 'y' | 'z';
  triangles: number;
  /** resources the model referenced that were not among the selected files */
  missing: string[];
}

export const MODEL_EXTS = ['glb', 'gltf', 'obj', 'stl'];
/** everything the file picker accepts: models plus their side files (buffers, materials, textures) */
export const UPLOAD_ACCEPT = '.glb,.gltf,.obj,.stl,.bin,.mtl,.png,.jpg,.jpeg,.webp,.gif,.bmp,.tga';

const ext = (name: string) => name.split('.').pop()?.toLowerCase() ?? '';
const baseName = (url: string) => {
  const clean = url.split(/[?#]/)[0];
  try {
    return decodeURIComponent(clean.split(/[\\/]/).pop() ?? '').toLowerCase();
  } catch {
    return (clean.split(/[\\/]/).pop() ?? '').toLowerCase();
  }
};

let draco: DRACOLoader | null = null;

/**
 * Load a user-supplied model into a merged triangle mesh. Pass the model file together with any
 * files it references (a .gltf's .bin and textures, an .obj's .mtl and textures); they are matched
 * by file name. Materials and textures are kept for the "Materials" shading mode.
 */
export async function loadModelFiles(files: File[]): Promise<LoadedModel> {
  const main = MODEL_EXTS.map((e) => files.find((f) => ext(f.name) === e)).find(Boolean);
  if (!main) throw new Error(`No model among the dropped files — use .glb, .gltf, .obj or .stl`);
  // side files are served to the three.js loaders through blob URLs keyed by file name
  const urls = new Map<string, string>();
  for (const f of files) if (f !== main) urls.set(f.name.toLowerCase(), URL.createObjectURL(f));
  const missing = new Set<string>();
  let active = 0;
  let idle: (() => void) | null = null;
  const manager = new THREE.LoadingManager();
  manager.setURLModifier((url) => {
    if (url.startsWith('data:') || url.startsWith('blob:')) return url;
    return urls.get(baseName(url)) ?? url;
  });
  // track outstanding loads so MTL textures (which load in the background) are waited for
  const start = manager.itemStart.bind(manager), end = manager.itemEnd.bind(manager), err = manager.itemError.bind(manager);
  manager.itemStart = (u: string) => { active++; start(u); };
  manager.itemEnd = (u: string) => { end(u); if (--active <= 0) idle?.(); };
  manager.itemError = (u: string) => { if (!u.startsWith('data:')) missing.add(u.startsWith('blob:') ? 'texture' : baseName(u)); err(u); };
  const settle = () => (active > 0 ? Promise.race([new Promise<void>((r) => (idle = r)), new Promise<void>((r) => setTimeout(r, 20000))]) : Promise.resolve());

  try {
    let obj: THREE.Object3D;
    let up: 'y' | 'z' = 'y';
    const e = ext(main.name);
    if (e === 'glb' || e === 'gltf') {
      const loader = new GLTFLoader(manager);
      loader.setMeshoptDecoder(MeshoptDecoder);
      // the glTF-only decoder that three.js bundles (fetched only when a file is Draco-compressed)
      if (!draco) draco = new DRACOLoader().setDecoderPath(DRACO_GLTF_CONFIG);
      loader.setDRACOLoader(draco);
      const data = e === 'glb' ? await main.arrayBuffer() : await main.text();
      const gltf = await new Promise<{ scene: THREE.Object3D }>((res, rej) => loader.parse(data, '', res, rej)).catch((err) => {
        const msg = String((err as Error)?.message ?? err);
        throw new Error(missing.size ? `${msg} — select the .gltf together with ${[...missing].join(', ')}` : msg);
      });
      obj = gltf.scene;
    } else if (e === 'obj') {
      const objLoader = new OBJLoader(manager);
      const mtlFile = files.find((f) => ext(f.name) === 'mtl');
      if (mtlFile) {
        const mtl = new MTLLoader(manager).parse(await mtlFile.text(), '');
        mtl.preload();
        objLoader.setMaterials(mtl);
      }
      obj = objLoader.parse(await main.text());
    } else {
      const g = new STLLoader().parse(await main.arrayBuffer());
      obj = new THREE.Mesh(g);
      up = 'z';
    }
    await settle();
    const raw = mergeObject(obj, { appearance: true });
    if (raw.indices.length === 0) throw new Error('No triangle meshes found in the file');
    return { name: main.name, raw, up, triangles: raw.indices.length / 3, missing: [...missing] };
  } finally {
    // images are decoded by now; the blob URLs are no longer needed
    for (const u of urls.values()) URL.revokeObjectURL(u);
  }
}

/** Single-file convenience wrapper. */
export function loadModelFile(file: File): Promise<LoadedModel> {
  return loadModelFiles([file]);
}
