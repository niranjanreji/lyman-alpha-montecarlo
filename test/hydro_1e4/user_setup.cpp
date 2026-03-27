/*
 * user_setup.cpp — physical problem definition for standalone RT runs.
 *
 * Coordinates (x, y, z) are in cm, centered on the domain origin.
 * Called once per cell by mc_grid.cpp at startup.
 * Not used in PLUTO-coupled builds (PLUTO provides grid data directly).
 *
 * Copy and modify this file for each problem.
 */

#include <cmath>
#include "common.h"

/* Total hydrogen number density n_H [cm^-3].
 * mc_grid.cpp computes n_HI = n_H * x_HI(T) using the ionization
 * fraction table (temperature_HI.dat). */
double user_density(double x, double y, double z) {
    double r = sqrt(x*x + y*y + z*z);
    return (r < 3e18) ? 5.0 : 0.0;
}

/* Gas temperature [K]. */
double user_temperature(double x, double y, double z) {
    return 1e4;
}

/* Bulk velocity [cm/s]. */
void user_velocity(double x, double y, double z,
                   double *vx, double *vy, double *vz) {
    *vx = 0.0;
    *vy = 0.0;
    *vz = 0.0;
}

/* Point sources: fill pos[i][3] and lum[i] for each source. */
void user_sources(int *n_sources, double pos[][3], double lum[]) {
    *n_sources = 1;
    pos[0][0] = 0.0;  pos[0][1] = 0.0;  pos[0][2] = 0.0;
    lum[0] = 1e50;
}

/* Convenience wrapper for standalone builds. */
Grid* load_user_grid() {
    Grid *grid = init_grid(user_sources);
    build_fields(grid, user_density, user_temperature,
                 user_velocity);
    return grid;
}
