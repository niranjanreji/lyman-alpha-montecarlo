# Makefile for 3D Monte Carlo Ly-alpha Radiative Transfer
# Niranjan Reji

# Compilers and flags
CXX ?= clang++
CXXFLAGS = -std=c++20 -O3 -fopenmp
LIBS = -lhdf5_cpp -lhdf5

# Default target
all: help

# CPU build
cpu:
	$(CXX) $(CXXFLAGS) -o mc.exe main.cpp $(LIBS)
	@echo CPU compilation successful!

# Clean
clean:
	-@rm -f mc.exe *.o
	@echo Clean complete.

# Help
help:
	@echo Available targets:
	@echo   cpu   - Build CPU version (mc.exe)
	@echo   clean - Remove build artifacts
	@echo   help  - Show this message

.PHONY: all cpu clean help
