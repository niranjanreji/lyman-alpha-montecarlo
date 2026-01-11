#include "common.h"

extern "C" {
    void RadiativeTransfer(
        double dt, 
        const double* rho,
        const double* vr, 
        const double* r,
        const double* pr,
        int n,
        double* out_force) 
    {
        static double mu_h = 1.2;      // change based on information / system?

        const string fname = "../input/grid.h5";
        static Grid* grid = nullptr;

        if (!grid) grid = load_grid(fname);

        // check if grid encompasses PLUTO's grid, if not, complain

        double Lx, Ly, Lz;
        Lx = grid->Lx;
        Ly = grid->Ly;
        Lz = grid->Lz;

        if (Lx*Lx + Ly*Ly + Lz*Lz < r[n-1]*r[n-1]) cerr << "PLUTO domain doesn't fit!" << endl;

        // modify the grid num density, velocity arrays to mirror PLUTO's rho, vr arrays

        for (int idx = 0; idx < grid->nx*grid->ny*grid->nz - 1; ++idx) {
            int iz = idx % grid->nz;
            int iy = (idx / grid->nz) % grid->ny;
            int ix = idx / (grid->ny * grid->nz);

            double x_center = grid->x_centers[ix];
            double y_center = grid->y_centers[iy];
            double z_center = grid->z_centers[iz];

            double rad = sqrt(x_center*x_center + y_center*y_center + z_center*z_center);

            int pluto_idx = lower_bound(r, r + n, rad) - r;
            pluto_idx = clamp(pluto_idx, 0, n-1);

            grid->hi[idx] = (rho[pluto_idx]) / (mu_h * m_p);
            grid->sqrt_temp[idx] = (uint16_t)lround(sqrt((pr[pluto_idx] * mu_h * m_p) / (rho[pluto_idx] * k)));

            if (rad == 0.0) {
                grid->vx[idx] = 0.0;
                grid->vy[idx] = 0.0;
                grid->vz[idx] = 0.0;
            }
            else {
                double rhat_x = x_center / rad;
                double rhat_y = y_center / rad;
                double rhat_z = z_center / rad;

                double v_radial = vr[pluto_idx];

                double vx = rhat_x * v_radial;
                double vy = rhat_y * v_radial;
                double vz = rhat_z * v_radial;

                grid->vx[idx] = vx;
                grid->vy[idx] = vy;
                grid->vz[idx] = vz;
            }
        }

        // inject new photons into p

        static Photons* p = new Photons();
        
        monte_carlo(*p, *grid, dt, 100, true);
    }
}