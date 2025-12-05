"""
Plot momentum_profile.txt produced by the monte-carlo code.
Also compute total radial momentum and momentum per photon.

Assumes momentum_profile.txt has two columns:
  r_center(cm)   momentum_sum_in_bin (g*cm/s)
and that spectrum.txt (optional) contains one x value per photon (one line per photon).
"""

import numpy as np
import os
import matplotlib.pyplot as plt

# Constants (cgs)
h = 6.62607015e-27        # erg s
nu_alpha = 2.466e15       # Hz (from your code)
c = 29979245800.0         # cm/s

# Photon momentum (single Ly-alpha photon) in cgs:
p_photon = h * nu_alpha / c

# Filenames
momfile = "momentum_profile.txt"
specfile = "spectrum.txt"

# Load momentum profile
if not os.path.exists(momfile):
    raise SystemExit(f"Error: {momfile} not found in current directory.")

data = np.loadtxt(momfile)
r = data[:, 0]            # bin center in cm
momentum_bins = data[:, 1]  # momentum sum in each bin (g*cm/s)

# Basic derived quantities
total_momentum = momentum_bins.sum()               # total radial momentum transferred (g*cm/s)

# Try to infer photon count from spectrum.txt if present
if os.path.exists(specfile):
    with open(specfile, "r") as f:
        photon_count = sum(1 for _ in f)
else:
    # fallback: ask user to fill in photon_count here if spectrum.txt is not available
    # (Set photon_count below if you know it, e.g. photon_count = 10000)
    photon_count = None

# Compute per-photon numbers if possible
if photon_count is not None and photon_count > 0:
    momentum_per_photon = total_momentum / photon_count
    momentum_per_photon_in_photon_momenta = momentum_per_photon / p_photon
else:
    momentum_per_photon = None
    momentum_per_photon_in_photon_momenta = None

# Cumulative radial momentum (outwards from center to R)
cum_mom = np.cumsum(momentum_bins)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

ax1.plot(r, momentum_bins, marker='o', linestyle='-', markersize=3)
ax1.set_ylabel("Momentum in bin (g·cm/s)")
ax1.set_yscale("symlog")            # handle wide dynamic ranges
ax1.grid(True, which='both', ls=':', alpha=0.6)
ax1.set_title("Radial momentum profile")

ax2.plot(r, cum_mom, marker='.', linestyle='-')
ax2.set_xlabel("Radius (cm)")
ax2.set_ylabel("Cumulative radial momentum (g·cm/s)")
ax2.grid(True, ls=':', alpha=0.6)

plt.tight_layout()
plt.show()

# Print summary
print("\n=== Summary ===")
print(f"Total radial momentum (sum over all bins): {total_momentum:.5e} g·cm/s")
print(f"Single Ly-alpha photon momentum p = h*nu/c = {p_photon:.5e} g·cm/s")

if momentum_per_photon is not None:
    print(f"Photon count (in {specfile}): {photon_count}")
    print(f"Momentum per photon (averaged): {momentum_per_photon:.5e} g·cm/s")
    print(f"Momentum per photon in units of p_photon: {momentum_per_photon_in_photon_momenta:.3f} × p_photon")
else:
    print(f"Photon count not found. Place the spectrum.txt file with one line per photon in the same folder,")
    print("or set photon_count variable inside the script and rerun to get per-photon numbers.")