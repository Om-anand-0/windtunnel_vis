# Architecture plan

This is the plan written before implementation. It is kept up to date as the code evolves.

## Overview

```
 ┌────────────┐   mesh (grid coords)   ┌──────────────┐   flags (u32/cell)   ┌───────────────┐
 │ voxelize/  │ ─────────────────────▶ │ GPU voxelizer│ ───────────────────▶ │ solver/ (LBM) │
 │ presets,   │                        │ 3-axis wind- │                      │ D2Q9 / D3Q19  │
 │ loaders    │                        │ ing + shell  │                      │ reg-BGK + LES │
 └────────────┘                        └──────────────┘                      └──────┬────────┘
                                                                                    │ f (SoA, ping-pong)
                        macro pass (ρ,u, EMA stats, momentum-exchange force, reductions)
                                                                                    ▼
          ┌──────────────────────────── field textures (rgba16float, 3D) ───────────────────┐
          │ vel = (ux,uy,uz,ρ)   vort = (ωx,ωy,ωz,Q)   stat = (ρ̄, TI, ūx, solid)           │
          └──────┬───────────────┬─────────────────┬──────────────────┬──────────────────────┘
                 ▼               ▼                 ▼                  ▼
          slice heatmap    surface Cp on mesh   particles/trails   streamlines, Q-iso raymarch
                                     render/ (WebGPU, one MSAA pass)
```

World space **is** lattice space (1 unit = 1 cell). The 2D mode is a volume of depth 1, so the
WebGPU renderer, particle tracer and streamline integrator are shared between 2D and 3D.

## Solver data layout

* Populations `f` are stored **structure-of-arrays**: `f[i * N + cell]`, `cell = x + nx*(y + ny*z)`,
  `f32`. Two buffers are ping-ponged (A→B, B→A) so each step is a single fused
  *pull-stream + collide* kernel with perfectly coalesced loads/stores along x.
* Obstacle flags: `u32` per cell (0 = fluid, 1 = vehicle solid).
* Running statistics for the time-averaged fields: `meanA = (ūx, ūy, ūz, ρ̄)` and
  `meanB = <u·u>` per cell (exponential moving average, updated once per frame).
* Memory at 256×128×128: 2 × 318 MB populations + 84 MB statistics + 3 × 32 MB textures ≈ 0.85 GB.

## Shader passes (per frame)

| pass | kind | notes |
|---|---|---|
| `stream_collide` × N | compute | pull from `x − cᵢ`, boundary handling, regularized BGK + Smagorinsky, sponge near outlet |
| `macro` | compute | ρ, u → `vel` texture; EMA stats; momentum-exchange force on boundary links; workgroup reduction of (F, max‖u‖, ρ_ref) |
| `reduce` | compute | single workgroup sums the per-workgroup partials → 32-byte result, copied to a mappable ring buffer |
| `derived` | compute | central differences on `vel` → vorticity, Q-criterion; turbulence intensity from stats |
| `particles` | compute | RK2 advection of 100k+ tracers through trilinear-filtered `vel`, trail ring buffer |
| `streamlines` | compute | RK2 integration of rake seeds through the instantaneous field |
| render | render | ground grid, tunnel box, mesh (lit or Cp), slice heatmap + iso-contours, Q iso-surface raymarch (writes `frag_depth`), streamlines, trails (additive) |

N (steps per frame) is adapted automatically to hold the frame-rate target, times the user's
sim-speed multiplier.

### Collision operator

Regularized BGK (Latt & Chopard): the non-equilibrium part of `f` is projected onto its second-order
Hermite moment `Π^neq` before relaxation, which filters the ghost modes that make plain BGK blow up
near τ → ½. A Smagorinsky eddy viscosity is added from the same `Π^neq` (closed form, no
finite differences):
`τ_eff = ½ (τ₀ + sqrt(τ₀² + 18√2 C_s² ‖Π^neq‖ / ρ))`.

*Hybrid at walls.* Testing showed that the regularized projection excites an odd–even mode on cells
that receive bounce-back populations (it grew into reverse flow along the moving belt). Cells with any
bounce-back link therefore use plain BGK; belt-adjacent cells additionally get τ ≥ 0.53. The floor is
**not** applied on the body, where it created a spurious Couette drag through the under-body gap.

### Boundaries

* inlet (x = 0): equilibrium at (ρ = 1, U)
* outlet (x = nx−1): zero-gradient on post-collision populations + viscous sponge zone
* top / sides: free-slip (specular reflection)
* ground: free-slip, no-slip (halfway bounce-back) or **moving** (bounce-back with wall momentum
  `+6 wᵢ ρ (cᵢ·u_w)`) for the rolling road. Belt cells within 2 cells of the body are held still
  (tyre contact patches) — the wheels do not rotate, and a belt sliding under a stationary tyre rams
  the stagnant wedge fluid into it (4× drag overshoot in tests)
* inlet absorbing layer: the first 3 % of the tunnel is blended towards the free-stream equilibrium
* vehicle: halfway bounce-back; the force is the momentum exchange `F = Σ 2 (fᵢ* − wᵢρ₀) cᵢ` over
  fluid→solid links (the reference state removes the absolute pressure on floor contact patches)

## Voxelization (GPU)

Mesh triangles (index + vertex buffers) and a model→grid matrix go to a compute pipeline:

1. **Ray-crossing counts along x, y and z.** One thread per triangle rasterizes the triangle's
   projection onto the plane perpendicular to each axis at cell-centre columns, computes the
   crossing coordinate, and `atomicAdd`s `±1` (sign of the normal) into the first cell above the
   crossing.
2. **Column scan.** A prefix sum along each axis gives the winding number at every cell centre
   (non-zero rule ⇒ union of overlapping closed parts works).
3. **Surface shell.** Every triangle is sampled at ≤ ½-cell spacing and marks its cells, so thin
   parts (wings, mirrors) never vanish.
4. **Vote.** `solid = (≥ 2 of 3 axis tests) OR shell`. The majority vote makes non-watertight
   meshes tolerable (a hole only corrupts one axis).

The 2D mode reuses the exact same kernels with `nz = 1` and the grid plane at the vehicle's centre
line, which produces the centre-plane slice. The WebGL2 fallback runs the identical algorithm on the
CPU (it only needs a 2D slice).

## Render pipeline

WebGPU mode draws everything in one 4×MSAA render pass with custom WGSL pipelines; Three.js
provides the loaders, geometry construction for the presets, math, the perspective camera and
OrbitControls. The WebGL2 fallback (2D only) renders with Three.js' `WebGLRenderer`; the solver
runs raw fragment-shader passes on the same GL context and the field textures are handed to
Three.js materials as `ExternalTexture`s.

## Backends

`App` (units, averaging, stability watchdog, rake, UI) is API-agnostic and talks to a `Backend`
(`src/backend/types.ts`):

| | WebGPU | WebGL2 fallback |
|---|---|---|
| solver | `SolverGPU` (WGSL compute, 2D + 3D) | `SolverGL` (fragment shaders + MRT, 2D) |
| voxelizer | `VoxelizerGPU` (compute + atomics) | `voxelizeCPU` (same algorithm) |
| renderer | `RendererGPU` (raw WGSL pipelines) | `RendererGL` (Three.js + raw GPGPU tracers) |
| force readback | mapAsync ring, ≤ 2 frames in flight | PBO + fence, polled |
| step tuner signal | `onSubmittedWorkDone` latency | frame interval vs. vsync |

Validation cases receive a `SolverFactory`, so the same checks run on either backend.

## Module map

* `src/solver/`  – lattice constants, WGSL/GLSL generators, `SolverGPU`, `SolverGL`
* `src/voxelize/` – presets, file loaders, mesh normalization, GPU + CPU voxelizers
* `src/render/`  – WebGPU renderer, WebGL fallback renderer, colormaps, camera rig
* `src/ui/`      – control panel, readouts, Cd chart, legend, validation panel, capture
* `src/analysis/` – unit conversion (lattice ↔ SI), aero coefficients, Strouhal estimator, validation cases
