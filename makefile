# Makefile for 3D Monte Carlo Ly-alpha Radiative Transfer
# Niranjan Reji

# Compilers and flags
CXX ?= clang++
CXXFLAGS = -std=c++20 -O3 -fopenmp
TARGET ?= mc.exe

# OS-dependent HDF5 libraries
ifeq ($(OS),Windows_NT)
    LIBS = -lhdf5_cpp -lhdf5
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        LIBS = -lhdf5_serial_cpp -lhdf5_serial
    else ifeq ($(UNAME_S),Darwin)
        LIBS = -lhdf5_cpp -lhdf5
    endif
endif

# Default target
all: help

# CPU build
cpu:
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp $(LIBS)
	@echo CPU compilation successful! Output: $(TARGET)

# Clean
clean:
	-@rm -f $(TARGET) mc.exe *.o
	@echo Clean complete.

# Help
help:
	@echo Available targets:
	@echo   cpu   - Build CPU version (mc.exe)
	@echo   clean - Remove build artifacts
	@echo   help  - Show this message

.PHONY: all cpu clean help
