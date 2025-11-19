# Makefile for 3D Monte Carlo Ly-alpha Radiative Transfer
# Niranjan Reji

# Compilers and flags
CXX = clang++
CXXFLAGS = -std=c++20 -O3 -fopenmp
NVCC = nvcc
NVCCFLAGS = -O3 -std=c++17 --expt-relaxed-constexpr
LIBS = -lhdf5_cpp -lhdf5 -lszip -lzlib

# Default target
all: help

# CPU build
cpu:
	$(CXX) $(CXXFLAGS) -o mc.exe main.cpp $(LIBS)
	@echo CPU compilation successful!

# GPU build - separate compilation to avoid NVCC parsing HDF5 headers
gpu: routines_cuda/hdf5_loader.o
	$(NVCC) $(NVCCFLAGS) -o mc_cuda.exe main.cu routines_cuda/hdf5_loader.o $(LIBS)
	@echo GPU compilation successful!

# Compile HDF5 loader separately with clang++
# Assumes CUDA_PATH environment variable is set (default CUDA installation does this)
routines_cuda/hdf5_loader.o: routines_cuda/hdf5_loader.cpp routines_cuda/hdf5_loader.h
	$(CXX) -c -O3 -std=c++17 -o routines_cuda/hdf5_loader.o routines_cuda/hdf5_loader.cpp

# Clean
clean:
	-@rm -f mc.exe mc_cuda.exe routines_cuda/*.o *.o
	@echo Clean complete.

# Help
help:
	@echo Available targets:
	@echo   cpu   - Build CPU version (mc.exe)
	@echo   gpu   - Build GPU version (mc_cuda.exe)
	@echo   clean - Remove build artifacts
	@echo   help  - Show this message

.PHONY: all cpu gpu clean help
