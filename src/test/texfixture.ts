import * as THREE from 'three';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';

/**
 * A small textured test car exported as a real .glb: a body with a canvas-texture livery,
 * transmissive glass, dark tyres and emissive lamps. Exercises the upload → materials path.
 */
export async function texturedCarGlb(): Promise<File> {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d')!;
  g.fillStyle = '#c81e1e';
  g.fillRect(0, 0, 256, 256);
  g.fillStyle = '#f5f5f5';
  for (let i = 0; i < 8; i++) g.fillRect(0, i * 32 + 10, 256, 8);
  g.fillStyle = '#111';
  g.font = 'bold 64px sans-serif';
  g.fillText('42', 90, 150);
  const livery = new THREE.CanvasTexture(c);
  livery.colorSpace = THREE.SRGBColorSpace;

  const root = new THREE.Group();
  const paint = new THREE.MeshStandardMaterial({ map: livery, metalness: 0.4, roughness: 0.3 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(4.2, 0.9, 1.8), paint);
  body.position.set(0, 0.75, 0);
  root.add(body);
  const glass = new THREE.MeshPhysicalMaterial({ color: 0x223344, transmission: 1, roughness: 0.05, metalness: 0 });
  const cabin = new THREE.Mesh(new THREE.BoxGeometry(2, 0.6, 1.6), glass);
  cabin.position.set(-0.2, 1.5, 0);
  root.add(cabin);
  const tyre = new THREE.MeshStandardMaterial({ color: 0x151515, roughness: 0.9 });
  for (const x of [-1.35, 1.35]) for (const z of [-0.85, 0.85]) {
    const w = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.38, 0.3, 24), tyre);
    w.rotation.x = Math.PI / 2;
    w.position.set(x, 0.38, z);
    root.add(w);
  }
  const lamp = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xfff2c0, emissiveIntensity: 2 });
  for (const z of [-0.6, 0.6]) {
    const l = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.15, 0.35), lamp);
    l.position.set(2.11, 0.9, z);
    root.add(l);
  }
  const glb = await new GLTFExporter().parseAsync(root, { binary: true });
  return new File([glb as ArrayBuffer], 'test-car.glb', { type: 'model/gltf-binary' });
}
