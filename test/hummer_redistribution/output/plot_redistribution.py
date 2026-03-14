import scipy
import subprocess
import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

# RII-A From Hummer 1962 - used for comparison in RASCAS
def analytic_redistribution(xo_arr, xi, sigma):
    prefac = pi**(-1.5)

    xo_arr = np.asarray(xo_arr, dtype=float)
    R = np.zeros_like(xo_arr)

    for i, xo in enumerate(xo_arr):
        u0 = 0.5 * abs(xo - xi)
        if xo > xi: 
            xbig = xo
            xsmall = xi
        else: 
            xbig = xi
            xsmall = xo

        def integrand(u):
            return np.exp(-u*u) * (np.arctan((xsmall + u)/sigma) - np.arctan((xbig - u)/sigma))
        val, _ = scipy.integrate.quad(integrand, u0, np.inf, epsabs=0, epsrel=1e-6, limit=200)
        R[i] = prefac*val
    
    area = np.trapezoid(R, xo_arr)
    return R / area

T = 100
a = 4.7e-4 / np.sqrt(T / 1e4)

xins = [0, 1, 2, 3, 4, 5, 8]
bins = np.linspace(-5, 13, 801)

fig, ax = plt.subplots(figsize=(10, 10), dpi=200)

for i, xin in enumerate(xins):
    fname = f"out_xin_{xin}.dat"
    data = np.loadtxt(fname)

    hist, be = np.histogram(data, bins=bins, density=True)
    bc = 0.5*(be[:-1] + be[1:])

    R_analytic = analytic_redistribution(bc, xin, a)
    analytic_label = "Analytic" if i == 0 else None
    ax.plot(bc, R_analytic, linestyle='--', color='black', linewidth=4.0, label=analytic_label)
    ax.plot(bc, hist, linewidth=1.8, alpha=0.95, label=f"x_in = {xin}")

ax.set_xlabel("x_out", fontsize=14)
ax.set_ylabel("PDF", fontsize=14)
ax.set_xlim(-5, 13)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=12)
ax.set_title("Single-Scatter Redistribution Function Test — T = 100 K (RII-A, Hummer 1962)", fontsize=14)
fig.tight_layout()
fig.savefig("redistribution.png", dpi=200, bbox_inches='tight')
plt.show()