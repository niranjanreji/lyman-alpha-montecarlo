# ------------------------------------------------------------------
# usage: python create_grid.py grid.h5
# creates static file with geometry and
# initial physical fields
# ------------------------------------------------------------------

import sys
import h5py
import datetime
import numpy as np

# ------------------------------------------------------------------
# the functions here are used by RASCSAS
# lya rate from recomb: ApJ 672:48-58, 2008
# case b recomb coeffi: MNRAS 292, 27-42, 1992
# excitation rate coef: MNRAS 407, 613-631, 2010
# ------------------------------------------------------------------

def lyalpha_rate_from_recomb(T):
    return 0.686 - 0.106*np.log(T/10**4) - 0.009*(T/10**4)**(-0.44)

def caseb_recomb_coeff(T):
    return 2.753e-14 * ((315614/T)**(1.5)) / ((1.0 + (315614/2.74)**(0.407))**(2.242))

def excitation_rate_coeff(T):
    return (2.41e-6/T**0.5) * (T/1e4)**0.22 * np.exp(-(10.2)/(8.617333262e-5 * T))


fname = sys.argv[1]

# grid resolution and domain (cm)
nx, ny, nz = 100, 100, 100
Lx, Ly, Lz = 6e18, 6e18, 6e18
dx, dy, dz = Lx/nx, Ly/ny, Lz/nz

x_edges = np.linspace(-Lx/2, Lx/2, nx+1)
y_edges = np.linspace(-Ly/2, Ly/2, ny+1)
z_edges = np.linspace(-Lz/2, Lz/2, nz+1)

x_centr = 0.5 * (x_edges[:-1] + x_edges[1:])
y_centr = 0.5 * (y_edges[:-1] + y_edges[1:])
z_centr = 0.5 * (z_edges[:-1] + z_edges[1:])

print("\nGrid Configuration")
print("-" * 70)
print(f"  Grid dimensions  : {nx} * {ny} * {nz} = {nx*ny*nz:,} cells")
print(f"  Cell spacing     : dx = {dx:.2e} cm")
print(f"                     dy = {dy:.2e} cm")
print(f"                     dz = {dz:.2e} cm")
print(f"  Domain size      : {Lx:.2e} * {Ly:.2e} * {Lz:.2e} cm^3")
print(f"  Cell volume      : {dx*dy*dz:.2e} cm^3")
print("-" * 70)

# create neutral hydrogen field
hi = np.zeros((nx, ny, nz))
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            r = np.sqrt(x_centr[i]**2 + y_centr[j]**2 + z_centr[k]**2)
            hi[i, j, k] = 5.0 if r < 3e18 else 0.0

# create ionization state grids
ne  = np.zeros((nx, ny, nz))
hii = np.zeros((nx, ny, nz))

# create temperature and velocity fields
vx = np.zeros((nx, ny, nz), dtype=np.float64)
vy = np.zeros((nx, ny, nz), dtype=np.float64)
vz = np.zeros((nx, ny, nz), dtype=np.float64)

sqrt_temp = np.full((nx, ny, nz), 1e2, dtype=np.int16)

# photon emission per cell using RASCAS formula
nphot = (dx*dy*dz * ne * (hii * lyalpha_rate_from_recomb(sqrt_temp**2)*caseb_recomb_coeff(sqrt_temp**2)
                           + hi * excitation_rate_coeff(sqrt_temp**2)))
total_grid_lum = np.sum(nphot)

# point source information (x, y, z, photons/sec)
ps_posx = [0]
ps_posy = [0]
ps_posz = [0]

ps_lumi = [1e50]

# write hdf5 file
with h5py.File(fname, "w") as f:
    # ------------------------------------------------------------------
    # /grid group: geometry only
    # ------------------------------------------------------------------

    g = f.create_group("grid")

    g.create_dataset("x_edges", data=x_edges)
    g.create_dataset("y_edges", data=y_edges)
    g.create_dataset("z_edges", data=z_edges)

    g.create_dataset("x_centers", data=x_centr)
    g.create_dataset("y_centers", data=y_centr)
    g.create_dataset("z_centers", data=z_centr)

    g.attrs["Lx"] = Lx
    g.attrs["Ly"] = Ly
    g.attrs["Lz"] = Lz
    g.attrs["nx"] = nx
    g.attrs["ny"] = ny
    g.attrs["nz"] = nz
    g.attrs["dx"] = dx
    g.attrs["dy"] = dy
    g.attrs["dz"] = dz
    g.attrs["units"] = "cgs"
    g.attrs["version"] = 1

    # ------------------------------------------------------------------
    # /fields group: physical cell data (static or initial)
    # ------------------------------------------------------------------

    fld = f.create_group("fields")

    fld.create_dataset("vx", data=vx, compression="gzip")
    fld.create_dataset("vy", data=vy, compression="gzip")
    fld.create_dataset("vz", data=vz, compression="gzip")

    #fld.create_dataset("ne", data=ne, compression="gzip")
    fld.create_dataset("n_HI", data=hi, compression="gzip")
    #fld.create_dataset("HII", data=hii, compression="gzip")

    fld.create_dataset("nphot", data=nphot, compression="gzip")
    fld.create_dataset("sqrt_temp", data=sqrt_temp, compression="gzip")

    fld.attrs["grid_luminosity"] = total_grid_lum

    # ------------------------------------------------------------------
    # /sources group: point source locations, information
    # ------------------------------------------------------------------

    src = f.create_group("sources")
    src.attrs["num"] = len(ps_posx)
    src.attrs["total_luminosity"] = sum(ps_lumi)

    src.create_dataset("ps_posx", data=ps_posx)
    src.create_dataset("ps_posy", data=ps_posy)
    src.create_dataset("ps_posz", data=ps_posz)

    src.create_dataset("ps_luminosity", data=ps_lumi)

    # ------------------------------------------------------------------
    # some metadata
    # ------------------------------------------------------------------

    f.attrs["creator"] = "grid_create.py"
    f.attrs["timestamp"] = datetime.datetime.now().isoformat()
    f.attrs["desc"] = "Input grid file for Ly-alpha RT"
