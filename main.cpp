/*
MAIN.CPP / NIRANJAN REJI
- UNITY BUILD: includes all modules for single-compilation-unit optimization
- PROGRAM ENTRY POINT
- COMMAND LINE ARGUMENT PARSING
- DATA LOADING AND SIMULATION INITIALIZATION
*/

#define _USE_MATH_DEFINES
#include <iostream>
#include "routines/common.h"

// Include all module .cpp files for unity build (after common.h)
#include "routines/model.cpp"
#include "routines/physics.cpp"
#include "routines/monte_carlo.cpp"

using namespace std;

// main(): program entry point
// loads CDF table and grid, runs Monte Carlo simulation
int main(int argc, char* argv[]) {
    // default parameters
    int n_photons = 100000;
    bool recoil  = true;
    string grid_path = "input/grid.h5";

    // parse command line arguments
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--photons" || arg == "-n") {
            if (i + 1 < argc) {
                n_photons = atoi(argv[++i]);
            }
        }
        else if (arg == "--no-recoil") {
            recoil = false;
        }
        else if (arg == "--grid") {
            if (i + 1 < argc) {
                grid_path = argv[++i];
            }
        }
        else if (arg == "--help" || arg == "-h") {
            cout << "Usage: " << argv[0] << " [options]\n";
            cout << "Options:\n";
            cout << "  -n, --photons N       Number of photons to simulate (default: 100000)\n";
            cout << "  --no-recoil           Disable recoil effect\n";
            cout << "  --grid PATH           Path to grid HDF5 file (default: input/grid.h5)\n";
            cout << "  -h, --help            Show this help message\n";
            return 0;
        }
    }

    cout << "=== 3D Monte Carlo Ly-alpha Radiative Transfer ===\n";
    cout << "Photons: " << n_photons << "\n";
    cout << "Recoil: " << (recoil ? "enabled" : "disabled") << "\n\n";

    // load grid
    cout << "Loading grid from " << grid_path << "...\n";
    try {
        load_grid(grid_path);
        cout << "  Grid dimensions: " << g_grid.nx << " x " << g_grid.ny << " x " << g_grid.nz << "\n";
        cout << "  Domain size: " << g_grid.Lx << " x " << g_grid.Ly << " x " << g_grid.Lz << " cm\n";
        cout << "  Cell size: " << g_grid.dx << " x " << g_grid.dy << " x " << g_grid.dz << " cm\n";
    } catch (const exception& e) {
        cerr << "Error loading grid: " << e.what() << "\n";
        return 1;
    }

    // run Monte Carlo simulation
    cout << "\nStarting Monte Carlo simulation...\n";
    monte_carlo(n_photons, recoil);

    cout << "\nSimulation complete!\n";
    return 0;
}
