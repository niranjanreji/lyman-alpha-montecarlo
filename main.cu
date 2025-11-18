/*
MAIN.CU / NIRANJAN REJI
- UNITY BUILD: includes all modules for single-compilation-unit optimization
- PROGRAM ENTRY POINT
- COMMAND LINE ARGUMENT PARSING
- DATA LOADING AND SIMULATION INITIALIZATION
*/

#include "routines_cuda/common.cuh"
#include "routines_cuda/model.cu"
#include "routines_cuda/physics.cu"
#include "routines_cuda/monte_carlo.cu"

using namespace std;

int main(int argc, char* argv[]) {
    // default parameters
    int n_photons = 100000;
    int threads = 256;
    int blocks = (n_photons + threads - 1)/threads;
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
    Grid3D grid;
    try {
        grid = load_grid(grid_path);
        cout << "  Grid dimensions: " << g_nx << " x " << g_ny << " x " << g_nz << "\n";
        cout << "  Domain size: " << g_nx*g_dx << " x " << g_ny*g_dy << " x " << g_nz*g_dz << " cm\n";
        cout << "  Cell size: " << g_dx << " x " << g_dy << " x " << g_dz << " cm\n";

        // run Monte Carlo simulation
        cout << "\nStarting Monte Carlo simulation...\n";
        monte_carlo_cuda(grid, n_photons, blocks, threads, recoil);

        cout << "\nSimulation complete!\n";
    return 0;
    } catch (const exception& e) {
        cerr << "Error loading grid: " << e.what() << "\n";
        return 1;
    }
}