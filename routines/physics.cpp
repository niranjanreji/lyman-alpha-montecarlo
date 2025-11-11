/*
PHYSICS.CPP / NIRANJAN REJI
- VOIGT FUNCTION APPROXIMATION (SMITH ET AL 2015)
- DISTANCE AS A FUNCTION OF TAU
- SCATTERING BEHAVIOR
*/

#include <cmath>
#include "common.h"
using namespace std;

// init_photon(): takes Photon, rng objects
// sets position, direction of photon
void init_photon(Photon& phot, xso::rng& rng, bool phi_symmetry) {

    phot.x = 0, phot.pos_x = 0, phot.pos_y = 0, phot.pos_z = 0;
    if (phi_symmetry)
    {
        uint64_t r1 = rng(); uint64_t r2 = rng();

        // shift by 11 bits, divide by max 53 bit val to convert to [0, 1) interval
        double u1 = double(r1 >> 11) * rng_const;
        double u2 = double(r2 >> 11) * rng_const;

        double cosine = u1*2.0 - 1.0;
        double sine   = sqrt(1.0 - cosine*cosine);
        double phi    = u2*2.0*pi;

        phot.dir_x = sine*cos(phi);
        phot.dir_y = sine*sin(phi);
        phot.dir_z = cosine;
        phot.phi = phi;
    }
    else
    {
        uint64_t r1 = rng(); uint64_t r2 = rng(); uint64_t r3 = rng();

        // shift by 11 bits, divide by max 53 bit val to convert to [0, 1) interval
        double u1 = double(r1 >> 11) * rng_const;
        double u2 = double(r2 >> 11) * rng_const;
        double u3 = double(r3 >> 11) * rng_const;

        double dir_x = u1*2.0 - 1.0;
        double dir_y = u2*2.0 - 1.0;
        double dir_z = u3*2.0 - 1.0;

        double dir_mag = sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z);
        phot.dir_x = dir_x / dir_mag;
        phot.dir_y = dir_y / dir_mag;
        phot.dir_z = dir_z / dir_mag;
    }
}

// voigt(x, T): takes temperature T, doppler frequency x
// returns H(x, T). uses approximation from COLT (2015)
inline double voigt(double x, int sqrt_T) {
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
inline double compute_t_to_boundary(Photon& phot, int ix, int iy, int iz) {
    double t_x = INFINITY, t_y = INFINITY, t_z = INFINITY;

    // distance to next x boundary
    if (phot.dir_x > 1e-10) t_x = (g_grid.x_edges[ix+1] - phot.pos_x) / phot.dir_x;
    else if (phot.dir_x < -1e-10) t_x = (g_grid.x_edges[ix] - phot.pos_x) / phot.dir_x;

    // distance to next y boundary
    if (phot.dir_y > 1e-10) t_y = (g_grid.y_edges[iy+1] - phot.pos_y) / phot.dir_y;
    else if (phot.dir_y < -1e-10) t_y = (g_grid.y_edges[iy] - phot.pos_y) / phot.dir_y;

    // distance to next z boundary
    if (phot.dir_z > 1e-10) t_z = (g_grid.z_edges[iz+1] - phot.pos_z) / phot.dir_z;
    else if (phot.dir_z < -1e-10) t_z = (g_grid.z_edges[iz] - phot.pos_z) / phot.dir_z;

    // return closest boundary, update cell indices
    return min(t_x, min(t_y, t_z));
}

// tau_to_s(): takes tau, photon
// returns next position of photon before scatter
void tau_to_s(double tau_target, Photon& phot) {
    // small constant to push past cell boundaries
    const double eps = g_grid.dx*1e-10;
    double tau_accumulated = 0.0;

    // define variables outside loop
    int ix, iy, iz, sqrt_T_local;
    double n_HI, vx_bulk, vy_bulk, vz_bulk, t_boundary, dir_dot_v, x_local, sigma_alpha, dtau, s, inv_sqrt_T;
    while (tau_accumulated < tau_target)
    {
        if (escaped(phot)) return;

        // current cell indices
        get_cell_indices(phot, ix, iy, iz);

        // clamp to valid values for safety
        ix = min(ix, g_grid.nx - 1);
        iy = min(iy, g_grid.ny - 1);
        iz = min(iz, g_grid.nz - 1);

        // get local cell properties
        sqrt_T_local = g_grid.sqrt_temp(ix, iy, iz);
        inv_sqrt_T   = 1.0 / sqrt_T_local; 
        n_HI    = g_grid.hi(ix, iy, iz);
        vx_bulk = g_grid.velx(ix, iy, iz);
        vy_bulk = g_grid.vely(ix, iy, iz);
        vz_bulk = g_grid.velz(ix, iy, iz);

        // find dist to next cell
        t_boundary = compute_t_to_boundary(phot, ix, iy, iz);

        // ensure minimum step to avoid getting stuck on boundaries
        if (t_boundary < eps) t_boundary = eps;

        // if no HI density, just move to next cell without accumulating optical depth
        if (n_HI < 1e-30) {
            phot.pos_x += (t_boundary + eps) * phot.dir_x;
            phot.pos_y += (t_boundary + eps) * phot.dir_y;
            phot.pos_z += (t_boundary + eps) * phot.dir_z;
            continue;
        }

        // find change in x due to local temperature
        phot.x               = phot.x * phot.local_sqrt_temp * inv_sqrt_T;
        phot.local_sqrt_temp = sqrt_T_local;

        // find local x due to bulk velocity
        dir_dot_v   = vx_bulk*phot.dir_x + vy_bulk*phot.dir_y + vz_bulk*phot.dir_z;
        x_local     = phot.x - (dir_dot_v * inv_sqrt_T) / (vth_const);
        sigma_alpha = 5.898e-14 * 1e2 * inv_sqrt_T * voigt(x_local, sqrt_T_local);

        // increment tau
        dtau = n_HI * sigma_alpha * (t_boundary + eps);

        // check if tau target is within current cell
        if (tau_accumulated + dtau >= tau_target) {
            s = (tau_target - tau_accumulated) / (n_HI * sigma_alpha);

            // update pos, return
            phot.pos_x += s*phot.dir_x;
            phot.pos_y += s*phot.dir_y;
            phot.pos_z += s*phot.dir_z;
            return;
        }

        // move to next cell
        tau_accumulated += dtau;
        phot.pos_x += (t_boundary + eps) * phot.dir_x;
        phot.pos_y += (t_boundary + eps) * phot.dir_y;
        phot.pos_z += (t_boundary + eps) * phot.dir_z;
    }
}

// u_parallel(): takes photon, rng objects, returns parallel atom velocity
// uses rejection method / CDF table / gaussian based on |x| regime
double u_parallel(double x_local, double sqrt_T_local, xso::rng& rng) {
    double x_abs = fabs(x_local);

    // x > 8 regime approximated by gaussian
    if (x_abs >= 8.0) 
    {
        // box-muller method
        uint64_t r1 = rng(); uint64_t r2 = rng();

        // shift by 11 bits, divide by max 53 bit val to convert to [0, 1) interval
        double u1 = double(r1 >> 11) * rng_const;
        double u2 = double(r2 >> 11) * rng_const;
        if (u1 == 0) u1 = 1.0;

        double z = sqrt(-2.0 * log(u1)) * cos(2.0*pi*u2);
        return (1.0 / x_local) + (z / sqrt(2.0));
    }

    // core approximated using rejection sampling
    //if (x_abs < 1.0)
    else
    {
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
        uint64_t R;
        while (true)
        {
            // draw univariate
            R  = rng();
            R1 = double(R >> 11) * rng_const;

            R = rng();
            r = double(R >> 11) * rng_const;
            // pick sampling regime
            if (R1 <= p) theta = -pi/2 + r*(theta0 - (-pi/2));
            else theta = theta0 + r*(pi/2 - theta0);

            u = a*tan(theta) + x_local;
            if (!isfinite(u)) continue;

            // draw second univariate
            R  = rng();
            R2 = double(R >> 11) * rng_const;

            // rejection method condition
            if ((R1 <= p) && (R2 <= exp(-u*u))) return u;
            if ((R1 > p) && (R2 <= exp(u02 - u*u))) return u;
        }
    }
}

// scatter_mu(): takes photon, rng
// returns mu = cos(theta) (scattering angle) from RASCAS distribution
double scatter_mu(double x_local, xso::rng& rng) {
    // generate univariate
    uint64_t r0 = rng();
    double r    = double(r0 >> 11) * rng_const;

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

    return pow(A+B, 1/3) - pow(A-B, 1/3);
}

// scatter(): takes photon, cell indices, radial momentum accumulator
// changes x, direction by scattering
// returns radial momentum transfer dp_r (positive = outward momentum to gas)
double scatter(Photon& phot, int ix, int iy, int iz, xso::rng& rng,
    bool recoil, bool phi_symmetry) {

    int sqrt_T_local = g_grid.sqrt_temp(ix, iy, iz);
    double vth       = vth_const * sqrt_T_local;
    double u_bulk_x  = g_grid.velx(ix, iy, iz) / vth;
    double u_bulk_y  = g_grid.vely(ix, iy, iz) / vth;
    double u_bulk_z  = g_grid.velz(ix, iy, iz) / vth;
    
    // shift photon x value to local frame
    double dirdotv  = u_bulk_x*phot.dir_x + u_bulk_y*phot.dir_y + u_bulk_z*phot.dir_z;
    double xlocal   = phot.x - dirdotv;
    
    // generate velocity components of scattering atom
    double u_paral = u_parallel(xlocal, sqrt_T_local, rng);
    
    // parallel components - sample normal using box-muller
    uint64_t r1 = rng(); uint64_t r2 = rng();

    double u1 = double(r1 >> 11) * rng_const;
    double u2 = double(r2 >> 11) * rng_const;
    if (u1 < 1e-16) u1 = 1e-16;
    
    double R = sqrt(-2.0*log(u1));
    double theta = two_pi*u2;

    double z1 = R*cos(theta);
    double z2 = R*sin(theta);

    double u_perp1 = sqrt_1_2*z1;
    double u_perp2 = sqrt_1_2*z2;

    // find basis that velocity components are in
    // a = random vector to cross with phot.dir
    double ax    = (fabs(phot.dir_x) < 0.9) ? 1.0 : 0.0;
    double ay    = (ax != 0.0) ? 0.0 : 1.0;

    // basis vector 2 (phot.dir = basis vector 1)
    double e1x = (-phot.dir_z*ay);
    double e1y = (phot.dir_z*ax);
    double e1z = ((phot.dir_x*ay) - (phot.dir_y*ax));

    // normalize e1
    double e1norm = sqrt(e1x*e1x + e1y*e1y + e1z*e1z);
    e1x /= e1norm; e1y /= e1norm; e1z /= e1norm;

    // basis vector 3
    double e2x = ((phot.dir_y*e1z) - (phot.dir_z*e1y));
    double e2y = ((phot.dir_z*e1x) - (phot.dir_x*e1z));
    double e2z = ((phot.dir_x*e1y) - (phot.dir_y*e1x)); 

    // find scattering angle, clamp cosine to [-1, 1] to avoid numerical issues
    double cosine = scatter_mu(xlocal, rng);
    cosine = max(-1.0, min(1.0, cosine));
    double sine = sqrt(1.0 - cosine*cosine);
    
    // draw univariate to sample phi
    // (reusing old declarations as a micro-optimization)
    r1 = rng();
    u1 = double(r1 >> 11) * rng_const;

    // pick new direction
    double phi;
    if (phi_symmetry) phi = phot.phi;
    else phi = u1 * 2 * pi;
    double cosphi = cos(phi), sinphi = sin(phi);

    // generate new direction vector
    double new_dir_x = cosine*phot.dir_x + sine*(cosphi*e1x + sinphi*e2x);
    double new_dir_y = cosine*phot.dir_y + sine*(cosphi*e1y + sinphi*e2y);
    double new_dir_z = cosine*phot.dir_z + sine*(cosphi*e1z + sinphi*e2z);

    // normalize just in case of numerical drift
    double invnorm = 1.0 / sqrt(new_dir_x*new_dir_x + new_dir_y*new_dir_y + new_dir_z*new_dir_z);
    new_dir_x *= invnorm; new_dir_y *= invnorm; new_dir_z *= invnorm;

    // dot product of (dimensionless) velocity and new direction
    double u_dot_k = new_dir_x*u_bulk_x + new_dir_y*u_bulk_y + new_dir_z*u_bulk_z;
    xlocal        += u_dot_k + u_paral*(cosine-1) + sine*(u_perp1*cosphi + u_perp2*sinphi);
    if (recoil) xlocal += 2.6e-4 * (1e2 / sqrt_T_local) * (cosine - 1);

    // calculate momentum transfer in radial direction
    // p_photon = (h*nu/c) * direction
    // momentum conservation: dp_gas = p_photon_in - p_photon_out

    // radial direction: r_hat = position / |position|
    double r_mag = sqrt(phot.pos_x*phot.pos_x + phot.pos_y*phot.pos_y + phot.pos_z*phot.pos_z);
    double r_hat_x = 0.0, r_hat_y = 0.0, r_hat_z = 0.0;

    if (r_mag > 1e-12) {
        r_hat_x = phot.pos_x / r_mag;
        r_hat_y = phot.pos_y / r_mag;
        r_hat_z = phot.pos_z / r_mag;
    }

    // -- faster approximate radial momentum (neglects O(vth/c) frequency term) --
    double A = hnualphabyc * (sqrt_T_local / 1e2);

    double dir_dot_r     = phot.dir_x * r_hat_x + phot.dir_y * r_hat_y + phot.dir_z * r_hat_z;
    double new_dir_dot_r = new_dir_x * r_hat_x + new_dir_y * r_hat_y + new_dir_z * r_hat_z;

    // dp_r ≈ A * (dir·r_hat - new_dir·r_hat)
    double dp_r = A * (dir_dot_r - new_dir_dot_r);

    // change frequency and direction
    phot.x     = xlocal;
    phot.dir_x = new_dir_x;
    phot.dir_y = new_dir_y;
    phot.dir_z = new_dir_z;

    return dp_r;
}