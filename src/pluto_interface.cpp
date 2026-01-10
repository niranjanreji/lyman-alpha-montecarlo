#include "common.h"

extern "C" {
    void RadiativeTransfer(
        double dt, 
        const double* rho,
        const double* vr, 
        const double* r,
        int n,
        double* out_force
    ) 
    {
        const string fname = "../input/grid.h5";

        Grid* grid = load_grid(fname);

        // check if grid encompasses PLUTO's grid, if not, complain

        double Lx, Ly, Lz;
        Lx = grid->Lx;
        Ly = grid->Ly;
        Lz = grid->Lz;

        if (Lx*Lx + Ly*Ly + Lz*Lz < r[n-1]*r[n-1]) cerr << "PLUTO domain doesn't fit!" << endl;

        // modify the grid num density array to mirror PLUTO's rho array

        for (int idx = 0; idx < grid->nx*grid->ny*grid->nz - 1; ++idx) {
            int iz = idx % grid->nz;
            int iy = (idx / grid->nz) % grid->ny;
            int ix = idx / (grid->ny * grid->nz);

            double x_center = grid->x_centers[ix];
            double y_center = grid->y_centers[iy];
            double z_center = grid->z_centers[iz];

            double rad = sqrt(x_center*x_center + y_center*y_center + z_center*z_center);
            
            double temp = 1e40;
            int pluto_idx = 0;
            for (int j = 0; j < n; ++j) {
                if (abs(rad - r[j]) < temp) {
                    temp = abs(rad - r[j]);
                    pluto_idx = j;
                }
            }

            grid->hi[idx] = rho[pluto_idx];
        }

        // modify the grid velocity arrays to mirror PLUTO's vr array

        // inject new photons into p

        static Photons* p = new Photons();
        
        monte_carlo(*p, *grid, dt, 100, true);
    }
}