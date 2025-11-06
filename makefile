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
$(TARGET): main.cpp model.cpp physics.cpp monte_carlo.cpp common.h
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(LIBS)
	@echo Compilation successful!

# Clean build artifacts
clean:
	-@rm -f $(TARGET) *.o
	@echo Clean complete.

# Rebuild everything
rebuild: clean all

# Run the simulation with default parameters
run: $(TARGET)
	./$(TARGET)

# Run with custom photon count (usage: make run-n N=1000000)
run-n: $(TARGET)
	./$(TARGET) -n $(N)

# Help target
help:
	@echo Available targets:
	@echo   all       - Build the executable (default)
	@echo   clean     - Remove build artifacts
	@echo   rebuild   - Clean and rebuild
	@echo   run       - Build and run with default parameters
	@echo   run-n     - Build and run with N photons (e.g., make run-n N=1000000)
	@echo   help      - Show this help message

.PHONY: all clean rebuild run run-n help
