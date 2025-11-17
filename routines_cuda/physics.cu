/*
PHYSICS.CPP / NIRANJAN REJI
- VOIGT FUNCTION APPROXIMATION (SMITH ET AL 2015)
- DISTANCE AS A FUNCTION OF TAU
- SCATTERING BEHAVIOR
*/

#include "common.cuh"

using namespace std;

// init_photon(): takes Photon, rng objects
// sets position, direction, frequency of photon
__device__ __forceinline__ void init_photon(Photon& phot, PhiloxState& rng_state, const Grid3D& grid) {

    phot.x = 0, phot.pos_x = 0, phot.pos_y = 0, phot.pos_z = 0;

    double dir_x = philox_uniform(rng_state)*2.0 - 1.0;
    double dir_y = philox_uniform(rng_state)*2.0 - 1.0;
    double dir_z = philox_uniform(rng_state)*2.0 - 1.0;

    double dir_mag = sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z);
    phot.dir_x = dir_x / dir_mag;
    phot.dir_y = dir_y / dir_mag;
    phot.dir_z = dir_z / dir_mag;
}

// voigt(x, T): takes temperature T, doppler frequency x
// returns H(x, T). uses approximation from COLT (2015)
__device__ __forceinline__ double voigt(double x, int sqrt_T) {
    double a = a_const / sqrt_T;
    double H = 0;
    double z = x*x;

    if (z >= 25)
    {
        H = 5 / (z - 5.5);
        H = z - 3.5 - H;
        H = z - 1.5 - (1.5/H);
        return (a/sqrt(pi)) / H;
    }

    double ez = exp(-z);

    if (z > 3)
    {
        H = B7 / (z - B8);
        H = z - B6 + H;
        H = z + B4 + (B5/H);
        H = z - B2 + (B3/H);
        H = B0 + (B1/H);
        return ez + a*H;
    }

    H = A5 / (z - A6);
    H = z - A4 + H;
    H = z - A2 + (A3/H);
    H = A0 + (A1/H);
    H = 1 - a*H;
    return ez*H;
}


// compute_t_to_boundary(): ray trace to next cell boundary
// takes photon object, current cell indices
// returns t such that next_pos = pos + t*dir crosses cell edge
__device__ __forceinline__ double compute_t_to_boundary(const Photon& p, const Grid3D& grid, int ix, int iy, int iz) {
    // fetch cell boundaries
    double x0 = tex1Dfetch<double>(grid.x_edges, ix);
    double x1 = tex1Dfetch<double>(grid.x_edges, ix+1);
    double y0 = tex1Dfetch<double>(grid.y_edges, iy);
    double y1 = tex1Dfetch<double>(grid.y_edges, iy+1);
    double z0 = tex1Dfetch<double>(grid.z_edges, iz);
    double z1 = tex1Dfetch<double>(grid.z_edges, iz+1);
    
    // find min distance to next cell along each axis, scaled by dir vector
    // using ternaries to minimize branchiness :)
    double t_x = (fabs(p.dir_x) > 1e-10) ? ( (p.dir_x > 0.0) ? (x1 - p.pos_x) / p.dir_x : (x0 - p.pos_x) / p.dir_x ) : INF;
    double t_y = (fabs(p.dir_y) > 1e-10) ? ( (p.dir_y > 0.0) ? (y1 - p.pos_y) / p.dir_y : (y0 - p.pos_y) / p.dir_y ) : INF;
    double t_z = (fabs(p.dir_z) > 1e-10) ? ( (p.dir_z > 0.0) ? (z1 - p.pos_z) / p.dir_z : (z0 - p.pos_z) / p.dir_z ) : INF;
    
    return fmin(t_x, fmin(t_y, t_z));
}

// tau_to_s(): takes tau, photon
// returns next position of photon before scatter
__device__ void tau_to_s(double tau_target, Photon& phot, const Grid3D& grid) {
    const double eps = g_dx*1e-10;
    double tau_accumulated = 0;

    // define variables outside loop
    int ix, iy, iz, sqrt_T_local, cell_idx;
    double n_hi, vx_bulk, vy_bulk, vz_bulk, step, dir_dot_v, x_local, sigma_alpha, dtau, s, inv_sqrt_T;
    while (tau_accumulated < tau_target)
    {
        if (escaped(phot, grid)) return;
        get_cell_indices(phot, grid, ix, iy, iz);

        ix = min(ix, g_nx - 1);
        iy = min(iy, g_ny - 1);
        iz = min(iz, g_nz - 1);

        cell_idx = ix*g_ny*g_nz + iy*g_nz + iz;

        // fetch local grid parameters
        n_hi    = tex1Dfetch<double>(grid.HI, cell_idx);
        vx_bulk = tex1Dfetch<double>(grid.vx, cell_idx);
        vy_bulk = tex1Dfetch<double>(grid.vy, cell_idx);
        vz_bulk = tex1Dfetch<double>(grid.vz, cell_idx);

        sqrt_T_local = tex1Dfetch<int>(grid.sqrt_T, cell_idx);
        inv_sqrt_T   = 1.0 / sqrt_T_local; 

        // dist to next cell 
        step = compute_t_to_boundary(phot, grid, ix, iy, iz);
        if (step < eps) step = eps;

        // if no HI density, just move to next cell without accumulating optical depth
        if (n_hi < 1e-30) {
            phot.pos_x += (step + eps) * phot.dir_x;
            phot.pos_y += (step + eps) * phot.dir_y;
            phot.pos_z += (step + eps) * phot.dir_z;
            continue;
        }

        // find change in x due to local temperature
        phot.x = phot.x * phot.local_sqrt_temp * inv_sqrt_T; 
        phot.local_sqrt_temp = sqrt_T_local;

        // find local x due to bulk velocity
        dir_dot_v   = vx_bulk*phot.dir_x + vy_bulk*phot.dir_y + vz_bulk*phot.dir_z;
        x_local     = phot.x - (dir_dot_v * inv_sqrt_T) / (vth_const);
        sigma_alpha = 5.898e-12 * inv_sqrt_T * voigt(x_local, sqrt_T_local);

        // increment tau
        dtau = n_hi * sigma_alpha * (step + eps);

        // check if tau target is within current cell
        if (tau_accumulated + dtau >= tau_target) {
            s = (tau_target - tau_accumulated) / (n_hi * sigma_alpha);

            // update pos, return
            phot.pos_x += s*phot.dir_x;
            phot.pos_y += s*phot.dir_y;
            phot.pos_z += s*phot.dir_z;
            return;
        }

        // move to next cell
        tau_accumulated += dtau;
        phot.pos_x += (step + eps) * phot.dir_x;
        phot.pos_y += (step + eps) * phot.dir_y;
        phot.pos_z += (step + eps) * phot.dir_z;
    }
}