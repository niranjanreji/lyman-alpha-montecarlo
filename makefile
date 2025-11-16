# Makefile for 3D Monte Carlo Ly-alpha Radiative Transfer
# Niranjan Reji

# Compiler and flags
CXX = clang++
CXXFLAGS = -std=c++23 -O3 -fopenmp

# Compiler settings (optimized for Windows/MSYS2)
LIBS = -lhdf5_cpp -lhdf5 -lszip -lzlib

# Target executable
TARGET = mc.exe

# Default target - direct compilation (no intermediate object files)
all: $(TARGET)

# Compile directly to executable (unity build)
$(TARGET): main.cpp routines/model.cpp routines/physics.cpp routines/monte_carlo.cpp routines/common.h
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(LIBS)
	@echo Compilation successful!

# Clean build artifacts
clean:
	-@rm -f $(TARGET) *.o
	@echo Clean complete.

# Rebuild everything
rebuild: clean all

# Help target
help:
	@echo Available targets:
	@echo   all       - Build the executable (default)
	@echo   clean     - Remove build artifacts
	@echo   rebuild   - Clean and rebuild
	@echo   help      - Show this help message

.PHONY: all clean rebuild run run-n help
