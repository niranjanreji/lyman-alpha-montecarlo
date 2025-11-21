# Lyman-Alpha Monte Carlo Radiative Transfer

A high-performance 3D Monte Carlo radiative transfer code for simulating Lyman-alpha photon propagation through hydrogen gas. Currently works on a Cartesian grid of uniform spacing, and is parallelized with OpenMP.

The code will be ported to CUDA soon. The eventual goal is to couple the code with hydrodynamics code to capture the effects of Lyman-alpha radiative pressure on galactic winds/outflows. Since line radiative transfer codes like this one take time on the order of ~a couple hours to converge upon observables we need, it is also necessary to speed up the code so it can keep up with the hydrodynamics computation - hence the goal to port it to CUDA.

(21/11): Current runtime ~ 1000 photons / 21 secs (16 processors) / 100^3 grid with n_hI = 5, T = 10^4 K
