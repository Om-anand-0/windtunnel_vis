import { dispatch1D } from '../gpu/device';
import type { SolverGPU } from '../solver/SolverGPU';
import type { MeshData } from './mesh';
import { crossingsWGSL, finalizeWGSL, projectWGSL, scanWGSL, surfaceWGSL } from './wgslVoxel';

export interface VoxelInfo {
  /** projected (frontal) area in cells² (3D) or projected height in cells (2D) */
  frontal: number;
  solidCells: number;
  min: [number, number, number];
  max: [number, number, number];
}

/** Voxelizes an indexed triangle mesh into a solver's flag buffer entirely on the GPU. */
export class VoxelizerGPU {
  private pipes: Record<string, GPUComputePipeline> = {};
  private meshKey: MeshData | null = null;
  private posBuf: GPUBuffer | null = null;
  private idxBuf: GPUBuffer | null = null;

  constructor(private device: GPUDevice) {
    const mk = (code: string, label: string) =>
      device.createComputePipeline({ label, layout: 'auto', compute: { module: device.createShaderModule({ code, label }), entryPoint: 'main' } });
    this.pipes.cross = mk(crossingsWGSL, 'vox-cross');
    this.pipes.scan = mk(scanWGSL, 'vox-scan');
    this.pipes.surf = mk(surfaceWGSL, 'vox-surface');
    this.pipes.fin = mk(finalizeWGSL, 'vox-finalize');
    this.pipes.proj = mk(projectWGSL, 'vox-project');
  }

  private upload(mesh: MeshData) {
    if (this.meshKey === mesh) return;
    this.posBuf?.destroy();
    this.idxBuf?.destroy();
    const d = this.device;
    this.posBuf = d.createBuffer({ size: Math.max(16, mesh.positions.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.idxBuf = d.createBuffer({ size: Math.max(16, mesh.indices.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.posBuf, 0, mesh.positions);
    d.queue.writeBuffer(this.idxBuf, 0, mesh.indices);
    this.meshKey = mesh;
  }

  /**
   * @param matrix column-major 4×4 transform from mesh space into grid space
   */
  async voxelize(solver: SolverGPU, mesh: MeshData | null, matrix: ArrayLike<number>): Promise<VoxelInfo> {
    const d = this.device;
    const { nx, ny, nz } = solver.dims;
    const n = solver.n;
    if (!mesh || mesh.indices.length === 0) {
      d.queue.writeBuffer(solver.flags, 0, new Uint32Array(n));
      return { frontal: 0, solidCells: 0, min: [0, 0, 0], max: [0, 0, 0] };
    }
    this.upload(mesh);
    const triCount = mesh.indices.length / 3;
    const hits = d.createBuffer({ size: 3 * n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const vote = d.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const info = d.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const readback = d.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(info, 0, new Uint32Array([0, 0, 0xffffffff, 0xffffffff, 0xffffffff, 0, 0, 0]));

    const uniforms: GPUBuffer[] = [];
    const mkParams = (axis: number, count: number) => {
      const disp = dispatch1D(count, 64);
      const buf = new ArrayBuffer(96);
      new Float32Array(buf, 0, 16).set(Array.from(matrix));
      const u = new Uint32Array(buf, 64, 8);
      u.set([nx, ny, nz, n, triCount, axis, disp.strideX, 0]);
      const ub = d.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(ub, 0, buf);
      uniforms.push(ub);
      return { ub, disp };
    };
    const bg = (pipe: GPUComputePipeline, entries: [number, GPUBuffer][]) =>
      d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: entries.map(([binding, buffer]) => ({ binding, resource: { buffer } })) });

    const enc = d.createCommandEncoder({ label: 'voxelize' });
    enc.clearBuffer(hits);
    enc.clearBuffer(vote);
    const pass = enc.beginComputePass();
    // 1. crossings
    {
      const { ub, disp } = mkParams(0, triCount);
      pass.setPipeline(this.pipes.cross);
      pass.setBindGroup(0, bg(this.pipes.cross, [[0, ub], [1, this.posBuf!], [2, this.idxBuf!], [3, hits]]));
      pass.dispatchWorkgroups(disp.x, disp.y);
    }
    // 2. scans along each axis
    const dimsArr = [nx, ny, nz];
    for (let a = 0; a < 3; a++) {
      const cols = dimsArr[(a + 1) % 3] * dimsArr[(a + 2) % 3];
      const { ub, disp } = mkParams(a, cols);
      pass.setPipeline(this.pipes.scan);
      pass.setBindGroup(0, bg(this.pipes.scan, [[0, ub], [3, hits], [4, vote]]));
      pass.dispatchWorkgroups(disp.x, disp.y);
    }
    // 3. surface shell
    {
      const { ub, disp } = mkParams(0, triCount);
      pass.setPipeline(this.pipes.surf);
      pass.setBindGroup(0, bg(this.pipes.surf, [[0, ub], [1, this.posBuf!], [2, this.idxBuf!], [4, vote]]));
      pass.dispatchWorkgroups(disp.x, disp.y);
    }
    // 4. vote → flags
    {
      const { ub, disp } = mkParams(0, n);
      pass.setPipeline(this.pipes.fin);
      pass.setBindGroup(0, bg(this.pipes.fin, [[0, ub], [4, vote], [5, solver.flags]]));
      pass.dispatchWorkgroups(disp.x, disp.y);
    }
    // 5. frontal projection + bbox
    {
      const { ub, disp } = mkParams(0, ny * nz);
      pass.setPipeline(this.pipes.proj);
      pass.setBindGroup(0, bg(this.pipes.proj, [[0, ub], [5, solver.flags], [6, info]]));
      pass.dispatchWorkgroups(disp.x, disp.y);
    }
    pass.end();
    enc.copyBufferToBuffer(info, 0, readback, 0, 32);
    d.queue.submit([enc.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const r = new Uint32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    for (const b of [hits, vote, info, readback, ...uniforms]) b.destroy();
    const ok = r[1] > 0;
    return {
      frontal: r[0],
      solidCells: r[1],
      min: ok ? [r[2], r[3], r[4]] : [0, 0, 0],
      max: ok ? [r[5] + 1, r[6] + 1, r[7] + 1] : [0, 0, 0],
    };
  }
}
