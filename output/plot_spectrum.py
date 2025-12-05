'''
# plot_spectrum.py
import numpy as np
import matplotlib.pyplot as plt

# Load output file (one x value per line)
x = np.loadtxt("spectrum.txt")
#y = np.loadtxt("spectrum (recoil on).txt")

# Make and show histogram
plt.figure(figsize=(6,4))
plt.hist(x, bins=200, density=True, color='royalblue', alpha=0.7)
#plt.hist(y, bins=200, density=True, color='yellow', alpha=0.5)
plt.xlabel("x (dimensionless frequency shift)")
plt.ylabel("Normalized counts")
plt.title("Emergent Lyα Spectrum")
plt.tight_layout()
plt.show()
'''

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
x = np.loadtxt("spectrum.txt")

# -----------------------------
# Ask user for parameters
# -----------------------------
tau0 = float(input("Enter line-center optical depth tau0: "))
a = float(input("Enter Voigt parameter a: "))

# -----------------------------
# Define analytic spectrum
# Dijkstra, Haiman & Spaans (2006), Eq. (9)
# -----------------------------
def J_sphere(x, a, tau0):
    # x: dimensionless frequency shift
    # a: Voigt parameter
    # tau0: line-center optical depth
    prefactor = np.sqrt(np.pi / 24.0) * (x**2) / (a * tau0)
    exponent = np.sqrt(2 * np.pi**3 / 27.0) * (np.abs(x)**3) / (a * tau0)
    return prefactor / (1 + np.cosh(exponent))

# -----------------------------
# Prepare x-grid for plotting the analytic curve
# -----------------------------
xmin, xmax = np.min(x), np.max(x)
x_plot = np.linspace(xmin, xmax, 2000)
J_vals = J_sphere(x_plot, a, tau0)

# Normalize analytic profile so area = 1 (PDF form)
J_vals /= np.trapz(J_vals, x_plot)

# -----------------------------
# Plot histogram + analytic curve
# -----------------------------
plt.figure(figsize=(6,4))
counts, bins, _ = plt.hist(
    x, bins=200, density=True,
    color='royalblue', alpha=0.7, label='Monte Carlo histogram'
)

plt.plot(
    x_plot, J_vals,
    color='black', linewidth=2,
    label='Analytic Lyα sphere profile'
)

plt.xlabel("x (dimensionless frequency shift)")
plt.ylabel("Normalized counts / PDF")
plt.title("Emergent Lyα Spectrum")
plt.legend()
plt.tight_layout()
plt.show()
