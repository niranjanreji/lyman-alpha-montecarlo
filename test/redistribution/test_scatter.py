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

print("Compiling test_scatter.cpp ...")
subprocess.run(["g++", "-std=c++20", "-O3", "test_scatter.cpp", "-o", "test_scatter", "-lhdf5_serial_cpp", "-lhdf5_serial"], check=True)

print("Running test_scatter ...")
subprocess.run(["./test_scatter"], check=True)

T = 100
a = 4.7e-4 / np.sqrt(T / 1e4)

xins = [0, 1, 2, 3, 4, 5, 8]
bins = np.linspace(-5, 13, 801)

plt.figure(figsize=(20,10))

for i, xin in enumerate(xins):
    fname = f"out_xin_{xin}.dat"
    data = np.loadtxt(fname)

    hist, be = np.histogram(data, bins=bins, density=True)
    bc = 0.5*(be[:-1] + be[1:])

    R_analytic = analytic_redistribution(bc, xin, a)
    analytic_label = "Analytic" if i == 0 else None
    plt.plot(bc, R_analytic, linestyle='--', color='black', label=analytic_label)
    plt.plot(bc, hist, label=f"x_in = {xin}")

plt.xlabel("x_out")
plt.ylabel("PDF")
plt.xlim(-5, 13)
plt.legend()
plt.title("Single-Scatter Redistribution Function Test")
plt.tight_layout()
plt.show()