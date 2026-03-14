# Lyman-alpha Monte Carlo Radiative Transfer

A Monte Carlo radiative transfer code for Lyman-alpha photons, designed for standalone use and coupling with the PLUTO hydrodynamics code.

## Overview

This code propagates Lyman-alpha photon packets through a 3D Cartesian grid, computing resonant scattering, momentum deposition, and energy exchange. It can run standalone (for benchmarking and testing) or coupled to PLUTO's 1D spherical hydro solver, where radiation forces and heating rates feed back into the gas dynamics.

Key features:
- Compile-time configuration via `rt_definitions.h`
- MPI parallelism (photon decomposition across ranks)
- OpenMP threading within each rank
- Multiple Voigt profile approximations
- Slab and full-box boundary geometries
- Save/restart of photon state across PLUTO output dumps
- Adaptive photon emission scaling with timestep

## Architecture

```
split_source.c          PLUTO calls SplitSource each hydro step
  |                       - gathers 1D radial arrays from all MPI ranks
  |                       - converts to CGS, fills LyaData struct
  v
pluto_interface.cpp     LyaRadiativeTransfer()
  |                       - interpolates 1D PLUTO data onto 3D RT grid
  |                       - calls monte_carlo()
  |                       - maps 3D momentum back to 1D radial bins
  v
monte_carlo.cpp         monte_carlo()
  |                       - emits photon packets, propagates them
  |                       - calls scatter() for each interaction
  v
physics.cpp             propagate(), scatter(), escaped()
                          - Voigt cross section, frequency redistribution
                          - momentum/energy deposition to grid
                          - boundary condition checks
```

For standalone runs, `main.cpp` replaces the PLUTO driver and calls the same RT functions using grid data from `user_setup.cpp`.

## Compile-time configuration

All RT options are set in `rt_definitions.h`, separate from PLUTO's `definitions.h`.

### Physics flags

| Flag | Options | Description |
|------|---------|-------------|
| `COUPLE_LYA_RT` | `TRUE` / `FALSE` | enable radiation force/energy coupling with hydro |
| `PHASE_FUNCTION` | `DIPOLE` / `ISOTROPIC` | scattering angle distribution |
| `VOIGT_FUNCTION` | `SMITH2015` / `HUMLICEK1982` / `TASITSIOMI2006` | Voigt profile approximation |
| `ENERGY_DEPOSIT` | `FDV` / `DIRECT` | heating rate method (see below) |
| `RECOIL` | `TRUE` / `FALSE` | include recoil in frequency redistribution |

### Energy deposition modes

- **FDV** (default): RT passes only momentum (F) to PLUTO. PLUTO computes heating as F dot v internally.
- **DIRECT**: RT tracks the energy deposited per scatter (h * delta_nu) and passes the heating rate directly.

### Domain and grid

| Flag | Description |
|------|-------------|
| `RTGEOMETRY` | `FULL_BOX` (all boundaries escape) or `SLAB` (periodic x/y, escape z) |
| `NX`, `NY`, `NZ` | grid resolution |
| `LX`, `LY`, `LZ` | domain size in cm |

### Photon emission

| Flag | Description |
|------|-------------|
| `N_PHOTONS` | reference packet count |
| `DT_PHOTONS` | reference timestep (s); set to a negative value to disable scaling |

When `DT_PHOTONS > 0`, the number of packets emitted per call scales as `n_emit = max(1, N_PHOTONS * dt / DT_PHOTONS)`, keeping packet weight constant across varying timesteps. When `DT_PHOTONS <= 0`, scaling is disabled and exactly `N_PHOTONS` packets are emitted every timestep.

### Output and diagnostics

| Flag | Description |
|------|-------------|
| `VERBOSE_OUTPUT` | detailed per-5% progress during MC loop |
| `SPECTRUM_OUTPUT` | write escaped photon frequencies to `output/spectrum.txt` |
| `POSITION_OUTPUT` | write photon position snapshots |
| `POSITION_INTERVAL` | scatter interval between position snapshots |
| `SMOOTHING` | experimental momentum smoothing |

## Problem setup (standalone)

For standalone runs, `user_setup.cpp` defines the physical problem:

```cpp
/* total hydrogen number density n_H [cm^-3] */
double user_density(double x, double y, double z) {
    double r = sqrt(x*x + y*y + z*z);
    return (r < 3e18) ? 5.0 : 0.0;
}

/* gas temperature [K] */
double user_temperature(double x, double y, double z) {
    return 1e4;
}

/* bulk velocity [cm/s] */
void user_velocity(double x, double y, double z,
                   double *vx, double *vy, double *vz) {
    *vx = *vy = *vz = 0.0;
}

/* point sources: luminosity in photons/s */
void user_sources(int *n_sources, double pos[][3], double lum[]) {
    *n_sources = 1;
    pos[0][0] = pos[0][1] = pos[0][2] = 0.0;
    lum[0] = 1e50;
}
```

The grid is built programmatically from these functions at startup. No external input files are needed.

## PLUTO coupling

In coupled mode, PLUTO's `SplitSource()` drives the RT each hydro step:

1. **Gather**: each MPI rank sends its local interior cells; rank 0 assembles the full 1D radial profile and broadcasts it.
2. **Interpolate**: `pluto_interface.cpp` maps the 1D spherical profile onto the 3D Cartesian RT grid using linear interpolation in radius.
3. **Monte Carlo**: photon packets are emitted, propagated, and scattered. Momentum and energy are deposited into the 3D grid.
4. **Map back**: 3D momentum densities are projected radially and binned onto PLUTO's 1D shells.
5. **Reduce**: `MPI_Allreduce` sums partial forces from all ranks.
6. **Apply**: `rhs.c` adds the radiation force to PLUTO's momentum equation and (optionally) the heating rate to the energy equation.

### Force-based timestep constraint

The coupling imposes a CFL-like constraint: the radiation force must not shift photon frequencies by more than a few thermal widths per step. The allowed velocity change is `dv_max = v_th * sqrt(ln(tau_drop))` where `tau_drop ~ 1e4`, giving `dv_max ~ 3 * v_th`. The timestep is then `dt = dv_max / (F/rho)`.

## MPI parallelization

- Every rank holds a full copy of the 3D grid.
- Photon packets are split across ranks (each emits `N_PHOTONS / nprocs`).
- No communication during the MC loop.
- After the loop: `MPI_Allreduce` on momentum/energy arrays.
- OpenMP parallelism within each rank for the photon loop.

## Building

### Standalone

```
make              # build standalone binary
make test         # build and run all tests
make clean
```

### PLUTO-coupled

The makefile integrates with PLUTO's build system. RT source files are compiled with `mpicxx` and linked alongside PLUTO's C objects. Key flags:
- `-DPARALLEL` from PLUTO's MPI configuration
- `-I./src -I.` for RT headers
- `-fopenmp` for OpenMP threading
- `-lstdc++` for C++ standard library

## File reference

| File | Purpose |
|------|---------|
| `rt_definitions.h` | compile-time RT configuration |
| `src/common.h` | shared constants, types, and declarations |
| `src/pluto_interface.h` | C-linkage interface between PLUTO and RT |
| `src/pluto_interface.cpp` | 1D-to-3D interpolation, force mapping, PLUTO entry point |
| `src/monte_carlo.cpp` | main MC loop, photon emission, diagnostics |
| `src/physics.cpp` | propagation, scattering, boundary conditions |
| `src/mc_grid.cpp` | grid construction from user functions |
| `src/photons.cpp` | photon emission and initialization |
| `src/voigt.cpp` | Voigt profile approximations |
| `user_setup.cpp` | problem-specific density, temperature, velocity, sources |
| `split_source.c` | PLUTO-side driver: data gathering, unit conversion, timestep logic |
| `rhs.c` | PLUTO RHS: applies radiation force and heating |
| `init.c` | PLUTO initialization and Analysis() diagnostics |
