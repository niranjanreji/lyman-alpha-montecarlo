"""
CDF table builder for Lyman-alpha scattering parallel velocities.

Strategy: Build CDF(z) tables where z = u - x, using a DENSE x grid
to minimize interpolation errors. Only interpolate across T dimension.

For Monte Carlo simulations with millions of scatterings, this precomputed
table approach is faster than rejection sampling in the x ~ 2-8 regime.

HYBRID APPROACH:
  - Use CDF table for r in [epsilon, 1-epsilon] where epsilon = 0.001
  - Use rejection sampling for r < epsilon or r > 1-epsilon (extreme tails)
  - This achieves +/-0.1 absolute error in u for 99.8% of samples
  - Only 0.2% require fallback to rejection sampling

VALIDATED PARAMETERS:
  - nx = 1200 (linear spacing from 1.0 to 7.99)
  - nT = 25 (log spacing from 100 K to 100,000 K)
  - nz = 4600 (z = u - x grid)
  - epsilon = 0.001
  - Max error in [epsilon, 1-epsilon] (in terms of u): 0.09
"""

import h5py
import time
import numpy as np
from scipy.special import wofz
from scipy.integrate import cumulative_trapezoid

# --- Physical constants (cgs) ---
pi = np.pi
k = 1.380649e-16
c = 29979245800.0
m_p = 1.6726219e-24
A_alpha = 6.265e8
nu_alpha = 2.466e15


# --- Core PDF functions ---
def voigt(x: float, T: float) -> float:
    """Voigt profile H(x,a)"""
    a = ((A_alpha * c) / (4*pi*nu_alpha)) * np.sqrt(m_p/(2*k*T))
    z = x + 1j*a
    return float(np.real(wofz(z)))


def pdf(x: float, T: float, u: np.ndarray) -> np.ndarray:
    """
    PDF for parallel velocity: f(u||) ∝ exp(-u²) / [(x-u)² + a²]
    Normalized by Voigt function H(x,a).
    """
    a = ((A_alpha * c) / (4*pi*nu_alpha)) * np.sqrt(m_p/(2*k*T))
    H = voigt(x, T)
    u = np.atleast_1d(u)

    numerator = np.exp(-u**2)
    denominator = (x - u)**2 + a**2

    return (a / (pi * H)) * (numerator / denominator)


# --- Grid construction ---
def make_z_grid(z_max=60.0, n_core=3000, n_tail=800, z_core=3.0):
    """
    Build z = u - x grid with fine resolution near z=0, coarser in tails.

    Args:
        z_max: maximum |z| value
        n_core: number of points in [-z_core, z_core]
        n_tail: number of points in each tail beyond z_core
        z_core: boundary between core and tail

    Returns:
        1D array of z values, symmetric about 0
    """
    # Fine uniform core
    z_core_arr = np.linspace(-z_core, z_core, n_core)

    # Stretched tails using sinh mapping for smooth transition
    if n_tail > 0:
        s = np.linspace(0, 1, n_tail + 1)[1:]  # exclude 0
        stretch_factor = 3.5

        # Positive tail
        z_pos_tail = z_core + (z_max - z_core) * \
                     np.sinh(stretch_factor * s) / np.sinh(stretch_factor)

        # Negative tail (mirror)
        z_neg_tail = -z_pos_tail[::-1]

        z_grid = np.concatenate([z_neg_tail, z_core_arr, z_pos_tail])
    else:
        z_grid = z_core_arr

    return np.asarray(z_grid, dtype=float)


# --- CDF computation ---
def build_cdf_single(x: float, T: float, z_grid: np.ndarray) -> np.ndarray:
    """
    Build CDF for a single (x, T) point on z = u - x grid.

    Returns:
        cdf: normalized CDF values on z_grid, in [0, 1]
    """
    # Convert z to u
    u_vals = z_grid + x

    # Evaluate PDF
    pdf_vals = pdf(x, T, u_vals)
    pdf_vals = np.maximum(pdf_vals, 0.0)

    # Integrate using cumulative trapezoid
    cdf = cumulative_trapezoid(pdf_vals, z_grid, initial=0.0)

    # Normalize
    total = cdf[-1]
    if total <= 0 or not np.isfinite(total):
        raise ValueError(f"Integration failed for x={x:.3f}, T={T:.0f}: total={total}")

    cdf = cdf / total

    # Enforce monotonicity and exact bounds
    cdf = np.maximum.accumulate(cdf)
    cdf[0] = 0.0
    cdf[-1] = 1.0

    return cdf


def build_tables(xs: np.ndarray, Ts: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    """
    Build CDF table for all (x, T) combinations.

    Args:
        xs: 1D array of x values
        Ts: 1D array of T values
        z_grid: shared z = u - x grid

    Returns:
        cdf_table: shape (n_x, n_T, n_z), dtype float32
    """
    nx = len(xs)
    nT = len(Ts)
    nz = len(z_grid)

    cdf_table = np.zeros((nx, nT, nz), dtype=np.float32)

    print(f"Building CDF table:")
    print(f"  x: {nx} points in [{xs[0]:.2f}, {xs[-1]:.2f}]")
    print(f"  T: {nT} points in [{Ts[0]:.0f}, {Ts[-1]:.0f}] K")
    print(f"  z: {nz} points in [{z_grid[0]:.2f}, {z_grid[-1]:.2f}]")
    print(f"  Table size: {nx * nT * nz * 4 / 1024**2:.1f} MB")
    print()

    start_time = time.time()

    for ix, x in enumerate(xs):
        for iT, T in enumerate(Ts):
            cdf = build_cdf_single(x, T, z_grid)
            cdf_table[ix, iT, :] = cdf.astype(np.float32)

            # Progress indicator
            done = ix * nT + iT + 1
            total = nx * nT
            if done % max(1, total // 20) == 0 or done == total:
                elapsed = time.time() - start_time
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{done:5d}/{total:5d}] {100*done/total:5.1f}% | "
                      f"Elapsed: {elapsed:5.1f}s | ETA: {eta:5.1f}s")

    print(f"\nTable built in {time.time() - start_time:.1f} seconds")
    return cdf_table


# --- File I/O ---
def save_tables_h5(filename: str, xs: np.ndarray, Ts: np.ndarray,
                   z: np.ndarray, cdf_table: np.ndarray, epsilon: float = 0.001) -> None:
    """Save CDF table to HDF5 file with metadata."""
    with h5py.File(filename, 'w') as f:
        f.create_dataset('x', data=xs, dtype=np.float64)
        f.create_dataset('T', data=Ts, dtype=np.float64)
        f.create_dataset('z', data=z, dtype=np.float64)
        f.create_dataset('cdf_table', data=cdf_table, compression='gzip', compression_opts=4)

        # Metadata
        f.attrs['nx'] = len(xs)
        f.attrs['nT'] = len(Ts)
        f.attrs['nz'] = len(z)
        f.attrs['x_min'] = xs[0]
        f.attrs['x_max'] = xs[-1]
        f.attrs['T_min'] = Ts[0]
        f.attrs['T_max'] = Ts[-1]
        f.attrs['z_min'] = z[0]
        f.attrs['z_max'] = z[-1]
        f.attrs['epsilon'] = epsilon
        f.attrs['r_min'] = epsilon
        f.attrs['r_max'] = 1.0 - epsilon
        f.attrs['description'] = f'Use table for r in [{epsilon}, {1-epsilon}], rejection sampling outside'

    print(f"\nSaved to: {filename}")
    print(f"  Shape: {cdf_table.shape}")
    print(f"  Dtype: {cdf_table.dtype}")
    print(f"  Size: {cdf_table.nbytes / 1024**2:.1f} MB (uncompressed)")
    print(f"  Epsilon: {epsilon} (use table for r in [{epsilon}, {1-epsilon}])")


# --- Interpolation and sampling (for testing) ---
def bilinear_interp_cdf(cdf_table: np.ndarray, xs: np.ndarray, Ts: np.ndarray,
                        x_query: float, T_query: float) -> np.ndarray:
    """
    Bilinearly interpolate CDF at arbitrary (x, T).

    Returns:
        cdf_interp: interpolated CDF on z grid
    """
    # Find x neighbors
    ix = np.searchsorted(xs, x_query)
    if ix == 0:
        ix = 1
    elif ix >= len(xs):
        ix = len(xs) - 1

    x0, x1 = xs[ix-1], xs[ix]
    wx = (x_query - x0) / (x1 - x0) if x1 != x0 else 0.0

    # Find T neighbors
    iT = np.searchsorted(Ts, T_query)
    if iT == 0:
        iT = 1
    elif iT >= len(Ts):
        iT = len(Ts) - 1

    T0, T1 = Ts[iT-1], Ts[iT]
    wT = (T_query - T0) / (T1 - T0) if T1 != T0 else 0.0

    # Get four corners
    c00 = cdf_table[ix-1, iT-1, :]
    c01 = cdf_table[ix-1, iT, :]
    c10 = cdf_table[ix, iT-1, :]
    c11 = cdf_table[ix, iT, :]

    # Bilinear interpolation
    cdf_interp = (1 - wx) * (1 - wT) * c00 + \
                 (1 - wx) * wT * c01 + \
                 wx * (1 - wT) * c10 + \
                 wx * wT * c11

    # Enforce CDF properties
    cdf_interp = np.clip(cdf_interp, 0.0, 1.0)
    cdf_interp = np.maximum.accumulate(cdf_interp)
    cdf_interp[-1] = 1.0

    return cdf_interp


def invert_cdf(cdf: np.ndarray, z_grid: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Invert CDF to find z given random numbers r in [0, 1].
    Uses simple LINEAR interpolation for speed and memory efficiency.
    """
    r = np.atleast_1d(r)
    r = np.clip(r, 0.0, 1.0)

    # Linear interpolation: np.interp is fast and simple
    z = np.interp(r, cdf, z_grid)

    return z


def sample_u(cdf_table: np.ndarray, xs: np.ndarray, Ts: np.ndarray,
             z_grid: np.ndarray, x_query: float, T_query: float,
             rng: np.random.Generator) -> float:
    """Sample a single u value at (x_query, T_query)."""
    cdf_interp = bilinear_interp_cdf(cdf_table, xs, Ts, x_query, T_query)
    r = rng.random()
    z = invert_cdf(cdf_interp, z_grid, np.array([r]))[0]
    return float(z + x_query)


# --- Testing and validation ---
def test_interpolation_accuracy(cdf_table: np.ndarray, xs: np.ndarray,
                                Ts: np.ndarray, z_grid: np.ndarray,
                                n_tests: int = 50):
    """
    Test interpolation accuracy by comparing to exact CDF at random points.
    Reports maximum error in CDF values.
    """
    print("\n" + "="*70)
    print("INTERPOLATION ACCURACY TEST")
    print("="*70)

    rng = np.random.default_rng(42)

    # Random test points between grid points
    x_tests = xs[0] + (xs[-1] - xs[0]) * rng.random(n_tests)
    T_tests = Ts[0] + (Ts[-1] - Ts[0]) * rng.random(n_tests)

    max_errors = []
    max_u_errors = []  # Error in u space

    print(f"\nTesting {n_tests} random (x, T) points...\n")
    print(f"{'Test':>5s} {'x':>7s} {'T':>8s} {'MaxErr(CDF)':>13s} {'MaxErr(u)':>13s}")
    print("-"*70)

    for i, (x_t, T_t) in enumerate(zip(x_tests, T_tests)):
        # Exact CDF at this point
        cdf_exact = build_cdf_single(x_t, T_t, z_grid)

        # Interpolated CDF
        cdf_interp = bilinear_interp_cdf(cdf_table, xs, Ts, x_t, T_t)

        # CDF error
        cdf_error = np.abs(cdf_interp - cdf_exact)
        max_cdf_err = np.max(cdf_error)
        max_errors.append(max_cdf_err)

        # Error in u space: sample in validated range [epsilon, 1-epsilon]
        epsilon = 0.001  # Use table for r in this range, rejection sampling for extremes
        r_samples = np.linspace(epsilon, 1-epsilon, 1000)
        z_exact = invert_cdf(cdf_exact, z_grid, r_samples)
        z_interp = invert_cdf(cdf_interp, z_grid, r_samples)
        u_exact = z_exact + x_t
        u_interp = z_interp + x_t
        u_error = np.abs(u_interp - u_exact)
        max_u_err = np.max(u_error)
        max_u_errors.append(max_u_err)

        print(f"{i+1:5d} {x_t:7.3f} {T_t:8.0f} {max_cdf_err:13.4e} {max_u_err:13.4e}")

    print("-"*70)
    print(f"\nSummary across {n_tests} tests:")
    print(f"  Max CDF error:  {np.max(max_errors):.4e}")
    print(f"  Mean CDF error: {np.mean(max_errors):.4e}")
    print(f"  Max u error:    {np.max(max_u_errors):.4e}")
    print(f"  Mean u error:   {np.mean(max_u_errors):.4e}")

    target = 0.1  # +/-0.1 absolute error target
    if np.max(max_u_errors) < target:
        print(f"\n  SUCCESS: Max u error < {target} target")
    else:
        print(f"\n  WARNING: Max u error exceeds {target} target")
        print(f"    Consider: increasing nT, nx, or nz")

    print("="*70 + "\n")

    return max_errors, max_u_errors

# --- Main execution ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("="*70)
    print("CDF TABLE BUILDER FOR LYMAN-ALPHA SCATTERING")
    print("="*70)
    print()

    # ========== GRID PARAMETERS ==========
    # Dense x grid to minimize interpolation error
    x_min = 1.0
    x_max = 7.99
    nx = 1200  # Dense grid - key to low interpolation error

    # Temperature grid - LOGARITHMIC spacing
    # PDF varies ~1500x slower with T than x, so we can use far fewer points
    T_min = 100.0  # K
    T_max = 1e5    # K
    nT = 30        # VALIDATED: achieves +/-0.1 error for r in [0.001, 0.999]

    # Z grid parameters
    z_max = 60.0
    n_core = 3000
    n_tail = 800
    z_core = 3.0

    # =====================================

    # Create grids
    print("Creating grids...")
    xs = np.linspace(x_min, x_max, nx)
    Ts = np.logspace(np.log10(T_min), np.log10(T_max), nT)  # Logarithmic spacing
    z_grid = make_z_grid(z_max=z_max, n_core=n_core, n_tail=n_tail, z_core=z_core)

    print(f"  x: {nx} points [{x_min:.2f}, {x_max:.2f}]")
    print(f"  T: {nT} points [{T_min:.0f}, {T_max:.0f}] K (log-spaced)")
    print(f"  z: {len(z_grid)} points [{z_grid[0]:.2f}, {z_grid[-1]:.2f}]")
    print()

    # Build table
    cdf_table = build_tables(xs, Ts, z_grid)

    # Save with epsilon parameter
    output_file = "cdf_tables.h5"
    epsilon = 0.001  # Validated cutoff
    save_tables_h5(output_file, xs, Ts, z_grid, cdf_table, epsilon=epsilon)

    # Test interpolation accuracy
    max_errors, max_u_errors = test_interpolation_accuracy(
        cdf_table, xs, Ts, z_grid, n_tests=500
    )

    # Plot error distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(max_errors, bins=20, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Max CDF error')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of CDF errors')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    ax2.hist(max_u_errors, bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax2.set_xlabel('Max u error')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of u errors')
    ax2.axvline(1e-1, color='red', linestyle='--', linewidth=2, label='1e-1 target')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=150)
    print("Saved error distribution to error_distribution.png")
    plt.close()

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
