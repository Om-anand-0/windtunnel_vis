import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { mergeObject } from './mesh';

export interface LoadedModel {
  name: string;
  raw: { positions: Float32Array; normals: Float32Array; indices: Uint32Array };
  /** best guess for the up axis based on file conventions */
  up: 'y' | 'z';
  triangles: number;
}

/** Load a user-supplied .glb/.gltf/.obj/.stl file into a merged triangle mesh. */
export async function loadModelFile(file: File): Promise<LoadedModel> {
  const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
  let obj: THREE.Object3D;
  let up: 'y' | 'z' = 'y';
  if (ext === 'glb' || ext === 'gltf') {
    const loader = new GLTFLoader();
    const data = ext === 'glb' ? await file.arrayBuffer() : await file.text();
    const gltf = await new Promise<{ scene: THREE.Object3D }>((res, rej) => loader.parse(data, '', res, rej));
    obj = gltf.scene;
  } else if (ext === 'obj') {
    obj = new OBJLoader().parse(await file.text());
  } else if (ext === 'stl') {
    const g = new STLLoader().parse(await file.arrayBuffer());
    obj = new THREE.Mesh(g);
    up = 'z';
  } else {
    throw new Error(`Unsupported file type ".${ext}" — use .glb, .gltf, .obj or .stl`);
  }
  const raw = mergeObject(obj);
  if (raw.indices.length === 0) throw new Error('No triangle meshes found in the file');
  return { name: file.name, raw, up, triangles: raw.indices.length / 3 };
}
