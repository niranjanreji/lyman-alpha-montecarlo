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
    bool phi_sym = false;
    string cdf_table_path = "input/cdf_tables.h5";
    string grid_path = "input/grid.h5";

    // parse command line arguments
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--photons" || arg == "-n") {
            if (i + 1 < argc) {
                n_photons = atoi(argv[++i]);
            }
        }
        else if (arg == "--no-phi")
        {
            phi_sym = true;
        }
        else if (arg == "--no-recoil") {
            recoil = false;
        }
        else if (arg == "--cdf-table") {
            if (i + 1 < argc) {
                cdf_table_path = argv[++i];
            }
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
            cout << "  --no-phi              System is symmetric w.r.t phi\n";
            cout << "  --cdf-table PATH      Path to CDF table HDF5 file (default: input/cdf_table.h5)\n";
            cout << "  --grid PATH           Path to grid HDF5 file (default: input/grid.h5)\n";
            cout << "  -h, --help            Show this help message\n";
            return 0;
        }
    }

    cout << "=== 3D Monte Carlo Ly-alpha Radiative Transfer ===\n";
    cout << "Photons: " << n_photons << "\n";
    cout << "Recoil: " << (recoil ? "enabled" : "disabled") << "\n\n";

    // load CDF table
    cout << "Loading CDF table from " << cdf_table_path << "...\n";
    try {
        load_table(cdf_table_path);
        cout << "  x grid: " << g_table.nx << " points\n";
        cout << "  T grid: " << g_table.nT << " points\n";
        cout << "  z grid: " << g_table.nz << " points\n";
    } catch (const exception& e) {
        cerr << "Error loading CDF table: " << e.what() << "\n";
        return 1;
    }

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
    monte_carlo(n_photons, recoil, phi_sym);

    cout << "\nSimulation complete!\n";
    return 0;
}
