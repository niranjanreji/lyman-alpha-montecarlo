"""
plot_sphere.py - Compare Monte Carlo results against analytical solution
for uniform sphere at T = 10^4 K with tau0 = 10^5, 10^6, 10^7
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Physical constants and parameters
# -----------------------------
T = 1e4  # temperature in K
sqrt_temp = 100  # sqrt(T) in K^0.5

# Voigt parameter at T = 10^4 K
# a = Gamma / (4 * pi * Delta_nu_D)
# For Ly-alpha: a ≈ 4.7e-4 at T = 10^4 K
a = 4.7e-4

# Optical depths corresponding to each test case
tau_values = {
    "1e5.txt": 1e5,
    "1e6.txt": 1e6,
    "1e7.txt": 1e7,
}

# Colors for each optical depth
colors = {
    1e5: "#1f77b4",  # blue
    1e6: "#2ca02c",  # green
    1e7: "#d62728",  # red
}

# -----------------------------
# Analytical solution: Dijkstra, Haiman & Spaans (2006), Eq. (9)
# -----------------------------
def J_sphere(x, a, tau0):
    """
    Analytical emergent spectrum from a uniform sphere.
    x: dimensionless frequency shift
    a: Voigt parameter
    tau0: line-center optical depth
    """
    prefactor = np.sqrt(np.pi / 24.0) * (x**2) / (a * tau0)
    exponent = np.sqrt(2 * np.pi**3 / 27.0) * (np.abs(x)**3) / (a * tau0)
    return prefactor / (1 + np.cosh(exponent))

# -----------------------------
# Set up figure
# -----------------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

# -----------------------------
# Load data and plot for each optical depth
# -----------------------------
for filename, tau0 in tau_values.items():
    try:
        x_data = np.loadtxt(filename)
    except FileNotFoundError:
        print(f"Warning: {filename} not found, skipping...")
        continue

    color = colors[tau0]
    label_mc = rf"MC $\tau_0 = 10^{{{int(np.log10(tau0))}}}$"
    label_an = rf"Analytical $\tau_0 = 10^{{{int(np.log10(tau0))}}}$"

    # Plot histogram of Monte Carlo results
    counts, bins, _ = ax.hist(
        x_data, bins=150, density=True,
        color=color, alpha=0.4, label=label_mc,
        histtype='stepfilled', edgecolor=color, linewidth=1.2
    )

    # Generate analytical curve over relevant x range
    x_plot = np.linspace(-50, 50, 250)
    J_vals = J_sphere(x_plot, a, tau0)

    # Normalize analytical profile (area = 1)
    J_vals /= np.trapezoid(J_vals, x_plot)

    # Plot analytical solution
    ax.plot(
        x_plot, J_vals,
        color=color, linewidth=2, linestyle='--', label=label_an
    )

# -----------------------------
# Formatting
# -----------------------------
ax.set_xlabel(r"$x$ (dimensionless frequency shift)", fontsize=12)
ax.set_ylabel(r"$J(x)$ (normalized PDF)", fontsize=12)
ax.set_title(
    rf"Emergent Ly$\alpha$ Spectrum from Uniform Sphere ($T = 10^4$ K, $a = {a:.1e}$)",
    fontsize=13
)

# Reorder legend: group MC and analytical separately
handles, labels = ax.get_legend_handles_labels()
# Sort so MC entries come first, then analytical
mc_handles = [h for h, l in zip(handles, labels) if "MC" in l]
mc_labels = [l for l in labels if "MC" in l]
an_handles = [h for h, l in zip(handles, labels) if "Analytical" in l]
an_labels = [l for l in labels if "Analytical" in l]

ax.legend(
    mc_handles + an_handles, mc_labels + an_labels,
    loc='upper right', fontsize=10, framealpha=0.9
)

ax.set_xlim(-30, 30)
ax.tick_params(axis='both', labelsize=10)

plt.tight_layout()
plt.savefig("sphere_comparison.png", dpi=150)
plt.show()

print("Saved figure to sphere_comparison.png")
