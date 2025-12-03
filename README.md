# Lyman-Alpha Monte Carlo Radiative Transfer

A 3D Monte Carlo radiative transfer code for simulating Lyman-alpha photon propagation through hydrogen gas. Currently works on a Cartesian grid of uniform spacing, and is parallelized with OpenMP.

The documentation here is a little sparse since the code is in development, and will be expanded in the future.

An input file for the code can be generated using the python script in the \input directory. The input file is written as a H5 file, using HDF5. Number densities, source information and luminosities can be modified within the code. The makefile provided uses the clang++ compiler, and a compiled binary is also provided.

A CUDA port was created, but 1) hardware tested has low double throughput (commercial GPU) 2) workload is highly divergent. 
Working on using FP32s exclusively, gaining access to HPC GPUs to test, and prepping to link to a hydrodynamics code as well.

(21/11): Current runtime ~ 1000 photons / 21 secs (16 processors) / 100^3 cell grid with n_hI = 5, T = 10^4 K (tau ~ 10^6)
