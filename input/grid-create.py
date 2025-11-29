#!/usr/bin/env python3
"""
Grid generation script for 3D Monte Carlo Ly-alpha radiative transfer
Creates a constant-spacing 3D Cartesian grid input file
"""

import numpy as np
import h5py

def create_3d_grid():
    """
    Creates a uniform 3D Cartesian grid for Monte Carlo simulation.
    Also creates a temperature grid simultaneously.
    Prompts user for grid parameters and saves to HDF5 file.
    """

    print("=" * 60)
    print("3D Grid Generator for Ly-alpha Monte Carlo Simulation")
    print("=" * 60)

    # Get grid parameters from user
    print("\nEnter grid parameters:")
    print("-" * 40)

    # Number of cells in each dimension
    nx = int(input("Number of cells in x-direction: "))
    ny = int(input("Number of cells in y-direction: "))
    nz = int(input("Number of cells in z-direction: "))

    # Grid spacing in each dimension (in cm, CGS units)
    dx = float(input("Grid spacing in x-direction (cm): "))
    dy = float(input("Grid spacing in y-direction (cm): "))
    dz = float(input("Grid spacing in z-direction (cm): "))

    print("\n" + "-" * 40)
    print(f" Creating grid with {nx} x {ny} x {nz} cells")
    print(f" Grid spacing: dx={dx:.2e} cm, dy={dy:.2e} cm, dz={dz:.2e} cm")

    # Calculate domain size
    Lx = nx * dx
    Ly = ny * dy
    Lz = nz * dz

    print(f" Domain size: {Lx:.2e} x {Ly:.2e} x {Lz:.2e} cm^3")
    print("-" * 40)

    # Create cell-centered coordinate arrays
    # Grid is centered at origin
    x_edges = np.linspace(-Lx/2, Lx/2, nx + 1)
    y_edges = np.linspace(-Ly/2, Ly/2, ny + 1)
    z_edges = np.linspace(-Lz/2, Lz/2, nz + 1)

    # Cell centers
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    # Create a 3D temperature array
    print(f" Temperatures are set to a constant throughout the grid. Modify T_grid to change this")

    # Since it's so numerically useful, create a 3D sqrt(T) array
    sqrt_T_grid = np.zeros((nx, ny, nz))
    sqrt_T_grid[:,:,:] = 1e2

    # Create a 3D HI number density array
    print(f" HI number densities are set to represent a uniform sphere. Modify HI_grid to change this")
    HI_grid = np.zeros((nx, ny, nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                r = np.sqrt(x_centers[i]**2 + y_centers[j]**2 + z_centers[k]**2)
                HI_grid[i, j, k] = 5 if r < Lx/2 else 0
    
    # Create a 3D bulk velocity field array
    print(f" Bulk velocity is set to 0 throughout grid. Modify v_grid to change this")
    vx_grid = np.zeros((nx, ny, nz))
    vy_grid = np.zeros((nx, ny, nz))
    vz_grid = np.zeros((nx, ny, nz))
    vx_grid[:,:,:] = 0
    vy_grid[:,:,:] = 0
    vz_grid[:,:,:] = 0


    # Save to HDF5 file
    print("-" * 40)
    output_file = "grid.h5"

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


        # Grid type metadata
        f.attrs['grid_type'] = 'uniform_cartesian'
        f.attrs['coordinate_system'] = 'cartesian'
        f.attrs['units'] = 'cgs'
        f.attrs['description'] = '3D uniform Cartesian grid for Ly-alpha Monte Carlo'

    print(f"\nGrid successfully saved to '{output_file}'")
    print("=" * 60)

    # Print summary
    print("\nGrid Summary:")
    print(f"  Total cells: {nx * ny * nz:,}")
    print(f"  x range: [{x_edges[0]:.2e}, {x_edges[-1]:.2e}] cm")
    print(f"  y range: [{y_edges[0]:.2e}, {y_edges[-1]:.2e}] cm")
    print(f"  z range: [{z_edges[0]:.2e}, {z_edges[-1]:.2e}] cm")
    print(f"  Cell volume: {dx * dy * dz:.2e} cm^3")
    print("=" * 60)


if __name__ == "__main__":
    create_3d_grid()
