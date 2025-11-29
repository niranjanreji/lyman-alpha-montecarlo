"""
Grid generation script for 3D Monte Carlo Ly-alpha radiative transfer
Creates a constant-spacing 3D Cartesian grid input file
"""

import numpy as np
import h5py

# From ApJ 672:48-58, 2008 (Cantalupo et. al - used by RASCAS too)
def lyalpha_rate_from_recomb(T):
    return 0.686 - 0.106*np.log(T/10**4) - 0.009*(T/10**4)**(-0.44)

# From MNRAS 292, 27-42, 1992 (Hui, Gnedin - used by RASCAS too)
def caseb_recomb_coeff(T):
    return 2.753 * 10**(-14) * ((315614/T)**(1.5)) / ((1.0 + (315614/2.74)**(0.407))**(2.242))

# MNRAS 407, 613-631, 2010 (Goerdt, Dekel, et. al - used by RASCAS too)
def excitation_rate_coeff(T):
    return (2.41e-6/T**0.5) * (T/1e4)**0.22 * np.exp(-(10.2)/(8.617333262e-5 * T))

def create_3d_grid():
    """
    Creates a uniform 3D Cartesian grid for Monte Carlo simulation.
    Also creates a temperature grid simultaneously.
    Prompts user for grid parameters and saves to HDF5 file.
    """

    print("\n" + "=" * 70)
    print(" " * 10 + "3D Grid Generator for Ly-alpha Monte Carlo")
    print("=" * 70)

    # Get grid parameters from user
    print("\nGrid Parameters")
    print("-" * 70)

    # Number of cells in each dimension
    nx = int(input("  Number of cells in x-direction: "))
    ny = int(input("  Number of cells in y-direction: "))
    nz = int(input("  Number of cells in z-direction: "))

    # Grid spacing in each dimension (in cm, CGS units)
    dx = float(input("  Grid spacing in x-direction (cm): "))
    dy = float(input("  Grid spacing in y-direction (cm): "))
    dz = float(input("  Grid spacing in z-direction (cm): "))

    # Calculate domain size
    Lx = nx * dx
    Ly = ny * dy
    Lz = nz * dz

    print("\nGrid Configuration")
    print("-" * 70)
    print(f"  Grid dimensions  : {nx} * {ny} * {nz} = {nx*ny*nz:,} cells")
    print(f"  Cell spacing     : dx = {dx:.2e} cm")
    print(f"                     dy = {dy:.2e} cm")
    print(f"                     dz = {dz:.2e} cm")
    print(f"  Domain size      : {Lx:.2e} * {Ly:.2e} * {Lz:.2e} cm^3")
    print(f"  Cell volume      : {dx*dy*dz:.2e} cm^3")
    print("-" * 70)

    # Create cell-centered coordinate arrays
    # Grid is centered at origin
    x_edges = np.linspace(-Lx/2, Lx/2, nx + 1)
    y_edges = np.linspace(-Ly/2, Ly/2, ny + 1)
    z_edges = np.linspace(-Lz/2, Lz/2, nz + 1)

    # Cell centers
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    # Create physical field arrays
    print("\nPhysical Fields")
    print("-" * 70)

    # Temperature field (constant in this example)
    sqrt_T_grid = np.full((nx, ny, nz), 1e2)
    print(f"  Temperature      : constant T = {1e4:.1e} K (sqrt(T) = {1e2:.1e} K^0.5)")

    # HI number density field (uniform sphere in this example)
    HI_grid = np.zeros((nx, ny, nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                r = np.sqrt(x_centers[i]**2 + y_centers[j]**2 + z_centers[k]**2)
                HI_grid[i, j, k] = 5.0 if r < Lx/2 else 0.0
    print(f"  HI density       : uniform sphere, n_HI = 5 cm^-3")

    # Ionization state (no ionization in this example)
    ne_grid = np.zeros((nx, ny, nz))
    HII_grid = np.zeros((nx, ny, nz))
    print(f"  Ionization       : neutral (n_e = n_HII = 0)")

    # Photon emission rate using RASCAS formula
    nphot_grid = (dx * dy * dz * ne_grid *
                  (HII_grid * lyalpha_rate_from_recomb(sqrt_T_grid**2) *
                   caseb_recomb_coeff(sqrt_T_grid**2) +
                   HI_grid * excitation_rate_coeff(sqrt_T_grid**2)))
    total_grid_lum = np.sum(nphot_grid)
    print(f"  Grid emission    : {total_grid_lum:.2e} photons/s")

    # Point sources (modify as needed)
    point_sources = [
        (0.0, 0.0, 0.0, 1e50)  # central point source
    ]

    if point_sources:
        print(f"  Point sources    : {len(point_sources)} source(s)")
        for i, ps in enumerate(point_sources):
            print(f"                     [{i}] ({ps[0]:.1e}, {ps[1]:.1e}, {ps[2]:.1e}) "
                  f"→ {ps[3]:.2e} photons/s")

    # Bulk velocity field (zero in this example)
    print(f"  Bulk velocities  : zero (v_x = v_y = v_z = 0)")
    print("-" * 70)
    vx_grid = np.zeros((nx, ny, nz))
    vy_grid = np.zeros((nx, ny, nz))
    vz_grid = np.zeros((nx, ny, nz))

    # Save to HDF5 file
    print("\nSaving to HDF5")
    print("-" * 70)
    output_file = "grid.h5"
    print(f"  Output file      : {output_file}")

    with h5py.File(output_file, 'w') as f:
        # Grid dimensions
        f.create_dataset('nx', data=nx)
        f.create_dataset('ny', data=ny)
        f.create_dataset('nz', data=nz)

        # Grid spacing
        f.create_dataset('dx', data=dx)
        f.create_dataset('dy', data=dy)
        f.create_dataset('dz', data=dz)

        # Cell edges (for boundary checking)
        f.create_dataset('x_edges', data=x_edges)
        f.create_dataset('y_edges', data=y_edges)
        f.create_dataset('z_edges', data=z_edges)

        # Cell centers (for evaluation of physical quantities)
        f.create_dataset('x_centers', data=x_centers)
        f.create_dataset('y_centers', data=y_centers)
        f.create_dataset('z_centers', data=z_centers)

        # Domain size
        f.create_dataset('Lx', data=Lx)
        f.create_dataset('Ly', data=Ly)
        f.create_dataset('Lz', data=Lz)

        # Temperature, Densities, Velocity Field
        f.create_dataset('sqrt_T', data=sqrt_T_grid)
        f.create_dataset('HI', data=HI_grid)
        f.create_dataset('vx', data=vx_grid)
        f.create_dataset('vy', data=vy_grid)
        f.create_dataset('vz', data=vz_grid)

        # Photon production rates (per second)
        f.create_dataset('nphot', data=nphot_grid)

        # Point sources
        n_sources = len(point_sources)
        f.create_dataset('n_point_sources', data=n_sources)

        if n_sources > 0:
            # Separate arrays for each property
            ps_x = np.array([ps[0] for ps in point_sources], dtype=np.float64)
            ps_y = np.array([ps[1] for ps in point_sources], dtype=np.float64)
            ps_z = np.array([ps[2] for ps in point_sources], dtype=np.float64)
            ps_n = np.array([ps[3] for ps in point_sources], dtype=np.float64)

            f.create_dataset('point_source_x', data=ps_x)
            f.create_dataset('point_source_y', data=ps_y)
            f.create_dataset('point_source_z', data=ps_z)
            f.create_dataset('point_source_luminosity', data=ps_n)

        # Grid type metadata
        f.attrs['grid_type'] = 'uniform_cartesian'
        f.attrs['coordinate_system'] = 'cartesian'
        f.attrs['units'] = 'cgs'
        f.attrs['description'] = '3D uniform Cartesian grid for Ly-alpha Monte Carlo'

    print(f"  Status           : successfully written")
    print("-" * 70)

    # Print summary
    print("\nSummary")
    print("-" * 70)
    print(f"  Total cells      : {nx * ny * nz:,}")
    print(f"  x range          : [{x_edges[0]:.2e}, {x_edges[-1]:.2e}] cm")
    print(f"  y range          : [{y_edges[0]:.2e}, {y_edges[-1]:.2e}] cm")
    print(f"  z range          : [{z_edges[0]:.2e}, {z_edges[-1]:.2e}] cm")
    total_lum = total_grid_lum + sum([ps[3] for ps in point_sources])
    print(f"  Total luminosity : {total_lum:.2e} photons/s")
    print("=" * 70)
    print()


if __name__ == "__main__":
    create_3d_grid()
