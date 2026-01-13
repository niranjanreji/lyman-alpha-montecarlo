#include "common.h"

extern "C" {
    void RadiativeTransfer(
        double dt, 
        const double* rho,
        const double* vr, 
        const double* r,
        const double* pr,
        const double* vol,
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

        double Rmax = sqrt(Lx*Lx + Ly*Ly + Lz*Lz) * 0.5;
        if (Rmax < r[n-1]) cerr << "PLUTO domain doesn't fit!" << endl;

        // modify the grid num density, velocity arrays to mirror PLUTO's rho, vr arrays

        for (int idx = 0; idx < grid->nx*grid->ny*grid->nz; ++idx) {
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

        // map 3D cartesian momentum back to PLUTO's 1D radial bins
        // first, zero out the output force array
        for (int i = 0; i < n; ++i) {
            out_force[i] = 0.0;
        }

        // accumulate momentum from each 3D cell into the corresponding radial bin
        for (int idx = 0; idx < grid->nx * grid->ny * grid->nz; ++idx) {
            int iz = idx % grid->nz;
            int iy = (idx / grid->nz) % grid->ny;
            int ix = idx / (grid->ny * grid->nz);

            double x_center = grid->x_centers[ix];
            double y_center = grid->y_centers[iy];
            double z_center = grid->z_centers[iz];

            double rad = sqrt(x_center*x_center + y_center*y_center + z_center*z_center);

            // find corresponding PLUTO radial bin
            int pluto_idx = lower_bound(r, r + n, rad) - r;
            pluto_idx = clamp(pluto_idx, 0, n - 1);

            // project cartesian momentum onto radial direction
            if (rad > 0.0) {
                double rhat_x = x_center / rad;
                double rhat_y = y_center / rad;
                double rhat_z = z_center / rad;

                double mom_radial = grid->mom_x[idx] * rhat_x
                                  + grid->mom_y[idx] * rhat_y
                                  + grid->mom_z[idx] * rhat_z;

                // convert to a force per unit volume
                out_force[pluto_idx] += mom_radial / (dt * vol[pluto_idx]);
            }
        }

        // clear the grid momentum for the next timestep
        fill(grid->mom_x.begin(), grid->mom_x.end(), 0.0);
        fill(grid->mom_y.begin(), grid->mom_y.end(), 0.0);
        fill(grid->mom_z.begin(), grid->mom_z.end(), 0.0);
    }
}