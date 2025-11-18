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
__device__ void tau_to_s(const double tau_target, Photon& phot, const Grid3D& grid) {
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


// u_parallel(): takes photon, rng objects, returns parallel atom velocity
// uses rejection method / gaussian based on |x| regime
__device__ double u_parallel(const double x_local, const double sqrt_T_local, PhiloxState& rng_state) {
    double x_abs = fabs(x_local);

    // |x| > 8 regime approximated by gaussian
    if (x_abs >= 8.0) {
        return (1.0/x_local) + inv_sqrt_2*philox_normal(rng_state);
    }

    // rest is sampled w/ rejection sampling
    else {
        double a    = a_const / sqrt_T_local;
        double zeta = log10(a);

        double x2 = x_local * x_local;
        double x3 = x2 * x_local;
        double x4 = x3 * x_local;
        double x5 = x4 * x_local;
        double z2 = zeta * zeta;

        double u0 = 2.648963 + 2.014446*zeta + 0.351479*z2;
        u0 += x_local*(-4.058673 - 3.675859*zeta - 0.640003*z2);
        u0 += x2*(3.017395 + 2.117133*zeta + 0.370294*z2);
        u0 += x3*(-0.869789 - 0.565886*zeta - 0.096312*z2);
        u0 += x4*(0.110987 + 0.070103*zeta + 0.011557*z2);
        u0 += x5*(-0.005200 - 0.003240*zeta - 0.000519*z2);

        double theta0 = atan((u0 - x_local) / a);
        double u02 = u0*u0;
        double eu0 = exp(-u02);
        double p = (theta0 + pi/2)/( (1 - eu0)*theta0 + (1 + eu0)*pi/2 );

        // pre-define variables for inner loop
        double R1, theta, u, R2, r;
        while (true)
        {
            // draw univariate
            R1 = philox_uniform(rng_state);
            r = philox_uniform(rng_state);

            // pick sampling regime
            if (R1 <= p) theta = -pi/2 + r*(theta0 - (-pi/2));
            else theta = theta0 + r*(pi/2 - theta0);

            u = a*tan(theta) + x_local;
            if (!isfinite(u)) continue;

            // draw second univariate
            R2 = philox_uniform(rng_state);

            // rejection method condition
            if ((R1 <= p) && (R2 <= exp(-u*u))) return u;
            if ((R1 > p) && (R2 <= exp(u02 - u*u))) return u;
        }
    }
}


// scatter_mu(): takes photon, rng
// returns mu = cos(theta) (scattering angle) from RASCAS distribution
__device__ double scatter_mu(const double x_local, PhiloxState& rng_state) {
    double r = philox_uniform(rng_state);

    double A, B;
    if (fabs(x_local) < 0.2)
    {
        B = 6 * (2*r - 1);
        A = sqrt(B*B + 35.972972972972973);
    }
    else
    {
        B = 4*r - 2;
        A = sqrt(B*B + 1);
    }

    return pow(A+B, 1.0/3.0) - pow(A-B, 1.0/3.0);
}


// scatter(): takes photon, cell indices, radial momentum accumulator
// changes x, direction by scattering
__device__ void scatter(Photon& phot, const Grid3D& grid, const int ix, 
    const int iy, const int iz, PhiloxState& rng_state, const bool recoil = true) {
    
    int cell_idx = ix*g_ny*g_nz + iy*g_nz + iz;
    int sqrt_T_local = tex1Dfetch<double>(grid.sqrt_T, cell_idx);
    double vth = vth_const * sqrt_T_local;
    double u_bulk_x = tex1Dfetch<double>(grid.vx, cell_idx) / vth;
    double u_bulk_y = tex1Dfetch<double>(grid.vy, cell_idx) / vth;
    double u_bulk_z = tex1Dfetch<double>(grid.vz, cell_idx) / vth;

    // shift photon x value to local frame
    double dirdotv  = u_bulk_x*phot.dir_x + u_bulk_y*phot.dir_y + u_bulk_z*phot.dir_z;
    double xlocal   = phot.x - dirdotv;

    // generate velocity components of scattering atom
    double u_paral = u_parallel(xlocal, sqrt_T_local, rng_state);

    // sample other velocity components
    double u_perp1, u_perp2;
    philox_normal_pair(rng_state, u_perp1, u_perp2);

    // find basis vectors for atom's velocity frame
    double ax    = (fabs(phot.dir_x) < 0.9) ? 1.0 : 0.0;
    double ay    = (ax != 0.0) ? 0.0 : 1.0;

    // basis vector 2 (phot.dir = basis vector 1)
    double e1x = (-phot.dir_z*ay);
    double e1y = (phot.dir_z*ax);
    double e1z = ((phot.dir_x*ay) - (phot.dir_y*ax));

    // normalize e1 (use inverse for consistency and speed)
    double inv_e1norm = 1.0 / sqrt(e1x*e1x + e1y*e1y + e1z*e1z);
    e1x *= inv_e1norm; e1y *= inv_e1norm; e1z *= inv_e1norm;

    // basis vector 3
    double e2x = ((phot.dir_y*e1z) - (phot.dir_z*e1y));
    double e2y = ((phot.dir_z*e1x) - (phot.dir_x*e1z));
    double e2z = ((phot.dir_x*e1y) - (phot.dir_y*e1x)); 

    // find scattering angle, clamp cosine to [-1, 1] to avoid numerical issues
    double cosine = scatter_mu(xlocal, rng_state);
    cosine = max(-1.0, min(1.0, cosine));
    double sine = sqrt(1.0 - cosine*cosine);
    
    // draw univariate to sample phi
    // (reusing old declarations as a micro-optimization)
    double u1  = philox_uniform(rng_state);
    double phi = u1 * 2 * pi; 

    double s, c;
    sincos(phi, &s, &c);

    // generate new direction vector
    double new_dir_x = cosine*phot.dir_x + sine*(c*e1x + s*e2x);
    double new_dir_y = cosine*phot.dir_y + sine*(c*e1y + s*e2y);
    double new_dir_z = cosine*phot.dir_z + sine*(c*e1z + s*e2z);

    // normalize just in case of numerical drift
    double invnorm = 1.0 / sqrt(new_dir_x*new_dir_x + new_dir_y*new_dir_y + new_dir_z*new_dir_z);
    new_dir_x *= invnorm; new_dir_y *= invnorm; new_dir_z *= invnorm;

    // dot product of (dimensionless) velocity and new direction
    double u_dot_k = new_dir_x*u_bulk_x + new_dir_y*u_bulk_y + new_dir_z*u_bulk_z;
    xlocal        += u_dot_k + u_paral*(cosine-1) + sine*(u_perp1*c + u_perp2*s);
    if (recoil) xlocal += 2.6e-4 * (1e2 / sqrt_T_local) * (cosine - 1);

    // change frequency and direction
    phot.x     = xlocal;
    phot.dir_x = new_dir_x;
    phot.dir_y = new_dir_y;
    phot.dir_z = new_dir_z;
}