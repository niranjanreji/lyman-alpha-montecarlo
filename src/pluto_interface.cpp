#include "common.h"

extern "C" {
    void RadiativeTransfer(
        const double* rho,
        const double* vr, 
        const double* r,
        int n,
        double* out_force
    ) {
        const string fname = "../input/grid.h5";

        Grid* grid = load_grid(fname);
        static Photons* p = new Photons();
        
        monte_carlo(*p, *grid, 1e20, 100, true);
    }
}