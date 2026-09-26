/**
 * Lattice velocity sets. Directions are listed so that opp(i) is easy to derive and so the
 * WGSL/GLSL generators can emit fully unrolled, constant-folded code.
 */
export interface Lattice {
  name: 'D2Q9' | 'D3Q19';
  dim: 2 | 3;
  q: number;
  c: [number, number, number][];
  w: number[];
  opp: number[];
  /** index of the direction with the y component mirrored */
  mirrorY: number[];
  /** index of the direction with the z component mirrored */
  mirrorZ: number[];
  /** index of the direction with both y and z mirrored */
  mirrorYZ: number[];
}

function build(name: Lattice['name'], dim: 2 | 3, c: [number, number, number][], w: number[]): Lattice {
  const find = (v: [number, number, number]) => {
    const k = c.findIndex((d) => d[0] === v[0] && d[1] === v[1] && d[2] === v[2]);
    if (k < 0) throw new Error('lattice not closed under reflection');
    return k;
  };
  return {
    name,
    dim,
    q: c.length,
    c,
    w,
    opp: c.map((d) => find([-d[0], -d[1], -d[2]])),
    mirrorY: c.map((d) => find([d[0], -d[1], d[2]])),
    mirrorZ: c.map((d) => find([d[0], d[1], -d[2]])),
    mirrorYZ: c.map((d) => find([d[0], -d[1], -d[2]])),
  };
}

export const D2Q9 = build(
  'D2Q9',
  2,
  [
    [0, 0, 0],
    [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0],
    [1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0],
  ],
  [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
);

export const D3Q19 = build(
  'D3Q19',
  3,
  [
    [0, 0, 0],
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    [1, 1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, -1], [0, 1, 1], [0, -1, -1],
    [1, -1, 0], [-1, 1, 0], [1, 0, -1], [-1, 0, 1], [0, 1, -1], [0, -1, 1],
  ],
  [1 / 3, ...Array(6).fill(1 / 18), ...Array(12).fill(1 / 36)],
);

/** Pairs (a,b), a<=b, of the symmetric second-order tensor components for a dimension. */
export function tensorPairs(dim: 2 | 3): [number, number][] {
  return dim === 2
    ? [[0, 0], [0, 1], [1, 1]]
    : [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]];
}

/** Format a number as a WGSL/GLSL float literal. */
export function fl(v: number): string {
  if (Number.isInteger(v)) return v.toFixed(1);
  return v.toPrecision(9);
}
