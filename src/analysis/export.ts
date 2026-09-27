import type { FieldExport } from '../solver/types';

/** Download helper shared by all exports. */
export function download(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

export function stamp() {
  return new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
}

export function csv(header: string[], rows: (number | string)[][], comments: string[] = []): Blob {
  const lines = [...comments.map((c) => `# ${c}`), header.join(',')];
  for (const r of rows) lines.push(r.map((v) => (typeof v === 'number' ? (isFinite(v) ? +v.toPrecision(7) : '') : v)).join(','));
  return new Blob([lines.join('\n') + '\n'], { type: 'text/csv' });
}

/**
 * Legacy-VTK (binary, big-endian) STRUCTURED_POINTS file readable by ParaView / VisIt.
 * Velocities are normalised by U∞, vorticity by U∞/L, Q by U∞²/L²; coordinates are metres.
 */
export function vtk(f: FieldExport, o: { dx: number; U: number; L: number; rhoRef: number; title: string }): Blob {
  const [nx, ny, nz] = f.dims;
  const n = nx * ny * nz;
  const h = o.dx * f.stride;
  const parts: BlobPart[] = [];
  const text = (s: string) => parts.push(new TextEncoder().encode(s));
  const floats = (count: number, get: (i: number, k: number) => number, comps: number) => {
    const buf = new DataView(new ArrayBuffer(count * comps * 4));
    for (let i = 0; i < count; i++) for (let k = 0; k < comps; k++) buf.setFloat32((i * comps + k) * 4, get(i, k), false);
    parts.push(buf.buffer);
    text('\n');
  };
  const cp = (rho: number) => (2 * (rho - o.rhoRef)) / (3 * o.U * o.U);
  text(`# vtk DataFile Version 3.0\n${o.title.slice(0, 250)}\nBINARY\nDATASET STRUCTURED_POINTS\n`);
  text(`DIMENSIONS ${nx} ${ny} ${nz}\nORIGIN ${0.5 * o.dx} ${0.5 * o.dx} ${nz > 1 ? 0.5 * o.dx : 0}\nSPACING ${h} ${h} ${h}\nPOINT_DATA ${n}\n`);
  text('VECTORS velocity float\n');
  floats(n, (i, k) => f.vel[4 * i + k] / o.U, 3);
  text('VECTORS mean_velocity float\n');
  floats(n, (i, k) => f.mean[4 * i + k] / o.U, 3);
  text('VECTORS vorticity float\n');
  floats(n, (i, k) => (f.vort[4 * i + k] * o.L) / o.U, 3);
  text('SCALARS Cp float 1\nLOOKUP_TABLE default\n');
  floats(n, (i) => cp(f.vel[4 * i + 3]), 1);
  text('SCALARS mean_Cp float 1\nLOOKUP_TABLE default\n');
  floats(n, (i) => cp(f.mean[4 * i + 3]), 1);
  text('SCALARS Q_criterion float 1\nLOOKUP_TABLE default\n');
  floats(n, (i) => (f.vort[4 * i + 3] * o.L * o.L) / (o.U * o.U), 1);
  return new Blob(parts, { type: 'application/octet-stream' });
}
