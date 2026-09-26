import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export type CameraPreset = 'side' | 'top' | 'front' | 'free';

/**
 * Perspective camera + orbit controls in lattice (grid) coordinates, with animated presets.
 * Uses the WebGPU clip-space convention (z ∈ [0, 1]) when `webgpu` is set.
 */
export class CameraRig {
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  private anim: { p0: THREE.Vector3; p1: THREE.Vector3; t0: THREE.Vector3; t1: THREE.Vector3; start: number; dur: number } | null = null;
  private dims = { nx: 256, ny: 128, nz: 128 };
  preset: CameraPreset = 'free';

  constructor(dom: HTMLElement, webgpu: boolean) {
    this.camera = new THREE.PerspectiveCamera(35, 1, 1, 20000);
    if (webgpu) this.camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
    this.controls = new OrbitControls(this.camera, dom);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.screenSpacePanning = true;
    this.controls.addEventListener('start', () => {
      this.anim = null;
    });
  }

  setDomain(dims: { nx: number; ny: number; nz: number }) {
    this.dims = dims;
    this.camera.near = Math.max(dims.nx, dims.ny) * 0.01;
    this.camera.far = Math.max(dims.nx, dims.ny, dims.nz) * 20;
    this.camera.updateProjectionMatrix();
  }

  private target(): THREE.Vector3 {
    const { nx, ny, nz } = this.dims;
    return new THREE.Vector3(nx * 0.5, ny * (nz > 1 ? 0.3 : 0.5), nz > 1 ? nz / 2 : 0.5);
  }

  /** Distance needed to fit a box of half-extent (w, h) in the viewport. */
  private fitDistance(w: number, h: number): number {
    const fov = (this.camera.fov * Math.PI) / 180;
    const aspect = this.camera.aspect;
    const dh = h / Math.tan(fov / 2);
    const dw = w / (Math.tan(fov / 2) * aspect);
    return Math.max(dh, dw) * 1.08;
  }

  goTo(preset: CameraPreset, animate = true) {
    this.preset = preset;
    const { nx, ny, nz } = this.dims;
    const is2D = nz <= 1;
    const t = this.target();
    let p: THREE.Vector3;
    switch (preset) {
      case 'side':
        t.y = ny * 0.5;
        p = new THREE.Vector3(t.x, t.y, t.z + this.fitDistance(nx / 2, ny / 2));
        break;
      case 'top':
        t.y = 0;
        p = new THREE.Vector3(t.x, this.fitDistance(nx / 2, (is2D ? nx * 0.2 : nz) / 2) + ny, t.z + 0.001);
        break;
      case 'front':
        t.y = ny * 0.35;
        t.x = nx * 0.35;
        p = new THREE.Vector3(-this.fitDistance((is2D ? nx * 0.25 : nz) / 2, ny / 2) * 0.8, t.y, t.z);
        break;
      default: {
        const d = this.fitDistance(nx / 2, ny / 2) * (is2D ? 0.95 : 0.8);
        const dir = new THREE.Vector3(-0.55, 0.42, 0.72).normalize();
        if (is2D) dir.set(-0.25, 0.18, 0.95).normalize();
        p = t.clone().add(dir.multiplyScalar(d));
      }
    }
    if (!animate) {
      this.camera.position.copy(p);
      this.controls.target.copy(t);
      this.controls.update();
      return;
    }
    this.anim = { p0: this.camera.position.clone(), p1: p, t0: this.controls.target.clone(), t1: t, start: performance.now(), dur: 650 };
  }

  update(width: number, height: number) {
    const aspect = width / Math.max(height, 1);
    if (Math.abs(aspect - this.camera.aspect) > 1e-4) {
      this.camera.aspect = aspect;
      this.camera.updateProjectionMatrix();
    }
    if (this.anim) {
      const k = Math.min(1, (performance.now() - this.anim.start) / this.anim.dur);
      const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
      this.camera.position.lerpVectors(this.anim.p0, this.anim.p1, e);
      this.controls.target.lerpVectors(this.anim.t0, this.anim.t1, e);
      if (k >= 1) this.anim = null;
    }
    this.controls.update();
    this.camera.updateMatrixWorld();
  }

  matrices(): { viewProj: Float32Array; invViewProj: Float32Array; eye: [number, number, number] } {
    const vp = new THREE.Matrix4().multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse);
    const inv = vp.clone().invert();
    const e = this.camera.position;
    return { viewProj: new Float32Array(vp.elements), invViewProj: new Float32Array(inv.elements), eye: [e.x, e.y, e.z] };
  }

  /** World-space ray through a canvas pixel (CSS px). */
  ray(x: number, y: number, w: number, h: number): THREE.Ray {
    const ndc = new THREE.Vector2((x / w) * 2 - 1, -(y / h) * 2 + 1);
    const rc = new THREE.Raycaster();
    rc.setFromCamera(ndc, this.camera);
    return rc.ray;
  }

  /** Project a world point to CSS pixels. */
  project(p: THREE.Vector3, w: number, h: number): THREE.Vector2 {
    const v = p.clone().project(this.camera);
    return new THREE.Vector2(((v.x + 1) / 2) * w, ((1 - v.y) / 2) * h);
  }
}
