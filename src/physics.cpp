/* physics.cpp — photon propagation, resonant scattering, and
 * boundary condition checks. implements the core physics of
 * lyman-alpha radiative transfer including voigt cross sections,
 * frequency redistribution, and momentum/energy deposition.
 *
 * Niranjan Reji, Raman Research Institute, March 2026
 * assisted by Claude (Anthropic) */

#include "common.h"
#include <rt_definitions.h>

/**
 * @brief check whether a photon has escaped the simulation domain.
 *
 * for SLAB geometry, applies periodic boundary conditions in x/y
 * and checks for escape in z. for FULL_BOX, checks all boundaries.
 *
 * @param g   grid reference
 * @param p   photon to check (position may be wrapped for SLAB)
 * @param ix  cell index in x (updated if wrapped)
 * @param iy  cell index in y (updated if wrapped)
 * @param iz  cell index in z
 * @return true if photon has escaped
 */
bool escaped(Grid& g, Photon& p, int& ix, int& iy, int& iz) {
    #if RTGEOMETRY == SLAB
        if (p.pos_x < -LX/2.0) {
            p.pos_x += LX;
        }
        else if (p.pos_x > LX/2.0) {
            p.pos_x -= LX;
        }
        if (p.pos_y < -LY/2.0) {
            p.pos_y += LY;
        }
        else if (p.pos_y > LY/2.0) {
            p.pos_y -= LY;
        }

        ix = (int)((p.pos_x + LX/2.0) / g.dx);
        iy = (int)((p.pos_y + LY/2.0) / g.dy);
        p.cell_idx = NY * NZ * ix + NZ * iy + iz;

        if (fabs(p.pos_z) > LZ/2.0) {
            p.escaped = 1;
            return true;
        }
        return false;
    #else
            if (ix < 0 || ix >= NX ||
            iy < 0 || iy >= NY ||
            iz < 0 || iz >= NZ) {
            p.escaped = 1;
            return true;
        }
        return false;
    #endif
}

/* helper struct for compute_next_boundary */
struct Boundary {
    double t;             /* step size */
    int axis, sign;
};

/**
 * @brief compute the distance to the next cell boundary along the photon's direction.
 * @param phot  photon with position and direction
 * @param g     grid with cell edges
 * @param ix    current cell index in x
 * @param iy    current cell index in y
 * @param iz    current cell index in z
 * @return Boundary struct with distance, axis crossed, and direction sign
 */
inline Boundary compute_next_boundary(Photon& phot, const Grid& g,
                                      const int ix, const int iy, const int iz) {
    constexpr double eps = (double)1e-10;

    const double px = phot.pos_x;
    const double py = phot.pos_y;
    const double pz = phot.pos_z;

    const double dx = phot.dir_x;
    const double dy = phot.dir_y;
    const double dz = phot.dir_z;

    /* figure out direction signs (ends up +1 or -1) */
    const int sx = (dx > 0) - (dx < 0);
    const int sy = (dy > 0) - (dy < 0);
    const int sz = (dz > 0) - (dz < 0);

    /* index of voxel faces that photon points to */
    const int ix_face = ix + (sx > 0);
    const int iy_face = iy + (sy > 0);
    const int iz_face = iz + (sz > 0);

    /* distances to next axis boundary along each respective axis */
    const double tx = (fabs(dx) > eps) ? (g.x_edges[ix_face] - px) / dx : INF;
    const double ty = (fabs(dy) > eps) ? (g.y_edges[iy_face] - py) / dy : INF;
    const double tz = (fabs(dz) > eps) ? (g.z_edges[iz_face] - pz) / dz : INF;

    /* find minimum */
    Boundary b{tx, 0, sx};
    if (ty < b.t) b = {ty, 1, sy};
    if (tz < b.t) b = {tz, 2, sz};
    return b;
}

/**
 * @brief propagate a photon through the grid until it accumulates target_tau
 *        optical depth, escapes, or runs out of time.
 * @param target_tau     optical depth to accumulate before stopping
 * @param phot           photon to propagate (position/cell updated in place)
 * @param g              grid with physical fields
 * @param ix             cell index in x (updated in place)
 * @param iy             cell index in y (updated in place)
 * @param iz             cell index in z (updated in place)
 * @param dt             time budget for this monte carlo step [s]
 * @param hit_time_limit set to true if photon runs out of time
 */
void propogate(const double target_tau, Photon& phot, Grid& g, int& ix, int& iy, int& iz,
               const double dt, bool& hit_time_limit) {

    const double eps = g.dx*1e-14;
    double tau = 0.0;

    double sqrt_temp, n_HI, ux, uy, uz, t, dir_dot_v, x_local, sigma_alpha, dtau, s, inv_sqrt_temp;
    double delta_t, distance, remaining_time, remaining_distance;
    double a;
    Boundary b;

    int cell_idx = phot.cell_idx;

    while (tau < target_tau) {
        if (escaped(g, phot, ix, iy, iz)) return;
        cell_idx = phot.cell_idx;

        /* get local cell properties */
        ux   = g.ux[cell_idx];
        uy   = g.uy[cell_idx];
        uz   = g.uz[cell_idx];
        n_HI = g.nHI[cell_idx];
        sqrt_temp = g.sqrt_temp[cell_idx];
        inv_sqrt_temp = 1.0 / sqrt_temp;

        /* distance to next cell boundary */
        b = compute_next_boundary(phot, g, ix, iy, iz);
        t = b.t;

        if (t < eps) t = eps;

        if (n_HI < 1e-40) {
            distance = t + eps;
            delta_t = distance * inv_c;   /* propagation time */

            /* check propagation time */
            if (phot.time + delta_t > dt) {
                remaining_time = dt - phot.time;
                remaining_distance = remaining_time * c;

                phot.pos_x += remaining_distance * phot.dir_x;
                phot.pos_y += remaining_distance * phot.dir_y;
                phot.pos_z += remaining_distance * phot.dir_z;

                phot.time = dt; hit_time_limit = true;
                return;
            }

            phot.pos_x += distance * phot.dir_x;
            phot.pos_y += distance * phot.dir_y;
            phot.pos_z += distance * phot.dir_z;
            phot.time += delta_t;

            if (b.axis == 0) {
                ix += b.sign;
                cell_idx += b.sign * (NY * NZ);
            }
            else if (b.axis == 1) {
                iy += b.sign;
                cell_idx += b.sign * NZ;
            }
            else {
                iz += b.sign;
                cell_idx += b.sign;
            }
            phot.cell_idx = cell_idx;
            continue;
        }

        /* compute change in x due to local temp */
        phot.x *= (phot.local_sqrt_temp * inv_sqrt_temp);
        phot.local_sqrt_temp = sqrt_temp;

        /* find local x due to bulk v (ux/uy/uz already dimensionless) */
        dir_dot_v = ux*phot.dir_x + uy*phot.dir_y + uz*phot.dir_z;
        x_local   = phot.x - dir_dot_v;
        sigma_alpha = 5.898e-12 * inv_sqrt_temp;

        a = a_const * inv_sqrt_temp;

        #if VOIGT_FUNCTION == SMITH2015
            if (a >= 0.05) {
                sigma_alpha *= voigt_humlicek(x_local, a);  /* outside smith range */
            }
            else {
                sigma_alpha *= voigt_smith(x_local, a);
            }
        #elif VOIGT_FUNCTION == HUMLICEK1982
            sigma_alpha *= voigt_humlicek(x_local, a);
        #elif VOIGT_FUNCTION == TASITSIOMI2006
            sigma_alpha *= voigt_tasitsiomi(x_local, a);
        #else
            #error "Set VOIGT_FUNCTION in rt_definitions.h correctly"
        #endif

        /* compute optical depth step */
        dtau = n_HI * sigma_alpha * (t + eps);

        /* is target tau within current depth (< dtau) */
        if (tau + dtau > target_tau) {
            s = (target_tau - tau) / (n_HI * sigma_alpha);
            delta_t = s * inv_c;

            if (phot.time + delta_t > dt) {
                remaining_time = dt - phot.time;
                remaining_distance = remaining_time * c;

                phot.pos_x += remaining_distance * phot.dir_x;
                phot.pos_y += remaining_distance * phot.dir_y;
                phot.pos_z += remaining_distance * phot.dir_z;

                phot.time = dt; hit_time_limit = true;
                return;
            }

            phot.pos_x += s * phot.dir_x;
            phot.pos_y += s * phot.dir_y;
            phot.pos_z += s * phot.dir_z;
            phot.time += delta_t;
            return;
        }

        /* move to next cell */
        tau += dtau;
        distance = t + eps;
        delta_t = distance * inv_c;

        /* check time limit before moving to next cell */
        if (phot.time + delta_t > dt) {
            remaining_time = dt - phot.time;
            remaining_distance = remaining_time * c;

            phot.pos_x += remaining_distance * phot.dir_x;
            phot.pos_y += remaining_distance * phot.dir_y;
            phot.pos_z += remaining_distance * phot.dir_z;

            phot.time = dt; hit_time_limit = true;
            return;
        }

        phot.pos_x += distance * phot.dir_x;
        phot.pos_y += distance * phot.dir_y;
        phot.pos_z += distance * phot.dir_z;
        phot.time += delta_t;

        if (b.axis == 0) {
            ix += b.sign;
            cell_idx += b.sign * (NY * NZ);
        }
        else if (b.axis == 1) {
            iy += b.sign;
            cell_idx += b.sign * NZ;
        }
        else {
            iz += b.sign;
            cell_idx += b.sign;
        }
        phot.cell_idx = cell_idx;
    }
}

/**
 * @brief sample the parallel component of the scattering atom's velocity.
 *
 * uses the rejection method following RASCAS for |x| < 8, and a
 * gaussian approximation via box-muller for |x| >= 8.
 *
 * @param x          dimensionless frequency in the bulk frame
 * @param sqrt_temp  sqrt of local temperature [K^0.5]
 * @param rng        per-photon random number generator
 * @return parallel atom velocity component in thermal units
 */
double u_parallel(double x, double sqrt_temp, xso::rng& rng) {
    double sign = (x >= 0.0) ? 1 : -1;
    double xabs = fabs(x);

    /* x > 8 regime: gaussian approximation using box-muller */
    if (xabs >= 8.0) {
        double u1 = urand(rng);
        double u2 = urand(rng);
        if (u1 < 1e-16) u1 = 1e-16;

        double z = sqrt(-log(u1)) * cos(two_pi * u2);
        return sign * ((1.0 / xabs) + z);
    }
    /* rejection method otherwise */
    {
        double a    = a_const / sqrt_temp;
        double zeta = log(a) * inv_ln_10;

        double u0 = 0;
        if (xabs >= 0.6) {
            const double x  = xabs;
            const double x2 = x * x;
            const double x4 = x2 * x2;
            const double z2 = zeta * zeta;

            /* rewritten using estrin's scheme to help parallelize */
            const double p0 =
                (2.648963 + 2.014446*zeta + 0.351479*z2)
                + x*(-4.058673 - 3.675859*zeta - 0.640003*z2);

            const double p1 =
                (3.017395 + 2.117133*zeta + 0.370294*z2)
                + x*(-0.869789 - 0.565886*zeta - 0.096312*z2);

            const double p2 =
                (0.110987 + 0.070103*zeta + 0.011557*z2)
                + x*(-0.005200 - 0.003240*zeta - 0.000519*z2);

            u0 = p0 + p1 * x2 + p2 * x4;
        }

        double theta0 = atan((u0 - xabs) / a);
        double u02 = u0*u0;
        double eu0 = exp(-u02);
        double p   = (theta0 + pi/2) / ((1 - eu0)*theta0 + (1 + eu0)*pi/2);

        double R1, theta, u, R2, r;
        while (true) {
            R1 = urand(rng);
            r  = urand(rng);

            /* pick sampling regime */
            if (R1 <= p) theta = -pi/2 + r * (theta0 + pi/2);
            else         theta = theta0 + r * (pi/2 - theta0);

            u = a * tan(theta) + xabs;
            if (!std::isfinite(u)) continue;

            R2 = urand(rng);

            if ((R1 <= p) && (R2 <= exp(-u*u))) return u * sign;
            if ((R1 > p) && (R2 <= exp(u02 - u*u))) return u * sign;
        }
    }
}

/**
 * @brief sample the scattering angle cosine from the RASCAS dipole distribution.
 * @param x    dimensionless frequency in the atom frame
 * @param rng  per-photon random number generator
 * @return mu = cos(theta) of the scattering angle
 */
double scatter_angle(double x, xso::rng& rng) {
    double r = urand(rng);
    double A, B;

    if (fabs(x) < 0.2) {
        B = 6.0 * (2.0 * r - 1.0);
        A = sqrt(B*B + mu_const);
    }
    else {
        B = 4.0 * r - 2.0;
        A = sqrt(B*B + 1.0);
    }

    return std::cbrt(A + B) - std::cbrt(A - B);
}

/**
 * @brief perform a resonant scattering event.
 *
 * samples the scattering atom velocity, computes the new photon
 * direction and frequency via partial redistribution, and deposits
 * the momentum (and optionally energy) change onto the grid.
 *
 * @param phot  photon to scatter (direction, frequency updated in place)
 * @param g     grid with physical fields and momentum arrays
 * @param rng   per-photon random number generator
 */
void scatter(Photon& phot, Grid& g, xso::rng& rng) {
    int cell_idx = phot.cell_idx;

    double sqrt_temp = g.sqrt_temp[cell_idx];
    double vth = vth_const * sqrt_temp;

    double u_bulk_x = g.ux[cell_idx];
    double u_bulk_y = g.uy[cell_idx];
    double u_bulk_z = g.uz[cell_idx];

    /* transform photon freq to local bulk frame */
    double dx = phot.dir_x;
    double dy = phot.dir_y;
    double dz = phot.dir_z;

    double x = phot.x;

    double dir_dot_v = u_bulk_x*dx + u_bulk_y*dy + u_bulk_z*dz;
    double x_local   = x - dir_dot_v;

    /* sample velocity of scattering atom */
    double u_paral = u_parallel(x_local, sqrt_temp, rng);

    /* transform x_local to atom frame */
    x_local -= u_paral;

    /* sample perpendicular velocity components of
     * atom using box-muller method */
    double u1 = urand(rng);
    double u2 = urand(rng);
    if (u1 < 1e-16) u1 = 1e-16;

    double R = sqrt(-log(u1));
    double theta = two_pi * u2;

    double u_perp1 = R * cos(theta);
    double u_perp2 = R * sin(theta);

    /* find orthonormal basis for atom velocity decomposition */
    double ax = (fabs(dx) < 0.9) ? 1.0 : 0.0;
    double ay = (ax != 0.0) ? 0.0 : 1.0;

    /* basis vector 2 */
    double e2x = (-dz*ay);
    double e2y = (dz*ax);
    double e2z = (dx*ay) - (dy*ax);

    double e2_norm = 1.0 / sqrt(e2x*e2x + e2y*e2y + e2z*e2z);
    e2x *= e2_norm; e2y *= e2_norm; e2z *= e2_norm;

    /* basis vector 3 */
    double e3x = (dy*e2z) - (dz*e2y);
    double e3y = (dz*e2x) - (dx*e2z);
    double e3z = (dx*e2y) - (dy*e2x);

    /* normalize if needed */
    double e3_norm_sq = e3x*e3x + e3y*e3y + e3z*e3z;
    if (fabs(e3_norm_sq - 1.0) > 1e-12) {
        double e3_norm = 1.0 / sqrt(e3_norm_sq);
        e3x *= e3_norm; e3y *= e3_norm; e3z *= e3_norm;
    }

    /* sample scattering angle */
    double cosine;
    #if PHASE_FUNCTION == ISOTROPIC
        cosine = urand(rng)*2.0 - 1.0;
    #else
        cosine = scatter_angle(x_local, rng);
    #endif
    cosine = std::max(-1.0, std::min(1.0, cosine));
    double sine = sqrt(1.0 - cosine*cosine);

    /* sample azimuthal angle uniformly */
    double phi = urand(rng) * two_pi;
    double cosphi, sinphi;
    sincos(phi, &sinphi, &cosphi);

    /* generate new direction vector */
    double new_dx = cosine*dx + sine*(cosphi*e2x + sinphi*e3x);
    double new_dy = cosine*dy + sine*(cosphi*e2y + sinphi*e3y);
    double new_dz = cosine*dz + sine*(cosphi*e2z + sinphi*e3z);

    /* normalize in case of numerical drift */
    double norm = new_dx*new_dx + new_dy*new_dy + new_dz*new_dz;
    if (fabs(norm - 1.0) > 1e-12) {
        double norm = 1.0 / sqrt(norm);
        new_dx *= norm; new_dy *= norm; new_dz *= norm;
    }

    /* compute outgoing frequency in bulk frame */
    double u_dot_k = new_dx*u_bulk_x + new_dy*u_bulk_y + new_dz*u_bulk_z;
    x_local       += u_dot_k + u_paral*cosine + sine*(u_perp1*cosphi + u_perp2*sinphi);
    #if RECOIL == TRUE
        x_local += 2.6e-4 * (1e2 / sqrt_temp) * (cosine - 1);
    #endif

    /* deposit momentum change onto grid, scaled by packet weight */

    double nu_old = nu_alpha * (1 + (x*vth/c));
    double nu_new = nu_alpha * (1 + (x_local*vth/c));

    double w = phot.weight;
    double px = h_by_c * (nu_old*dx - nu_new*new_dx);
    double py = h_by_c * (nu_old*dy - nu_new*new_dy);
    double pz = h_by_c * (nu_old*dz - nu_new*new_dz);

    #pragma omp atomic
    g.mom_x[cell_idx] += w * px;
    #pragma omp atomic
    g.mom_y[cell_idx] += w * py;
    #pragma omp atomic
    g.mom_z[cell_idx] += w * pz;

    #if ENERGY_DEPOSIT == DIRECT
        #pragma omp atomic
        g.energy[cell_idx] += w * h_by_c * (nu_old - nu_new);
    #endif

    phot.dir_x = new_dx; phot.dir_y = new_dy; phot.dir_z = new_dz;
    phot.x = x_local;
}
