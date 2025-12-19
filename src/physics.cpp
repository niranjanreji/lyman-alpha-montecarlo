// ---------------------------------------
// physics.cpp - raytracing, scattering
// ---------------------------------------

#include "common.h"

inline bool escaped(Grid& g, Photon& p, int ix, int iy, int iz) {
    if (ix < 0 || ix >= g.nx ||
        iy < 0 || iy >= g.ny ||
        iz < 0 || iz >= g.nz) {
        p.escaped = 1;
        return true;
    }
    return false;
}

// voigt(x, t): takes sqrt_temp, doppler freq x
// returns H(x, T) using approx from COLT (2015)
inline Real voigt(Real x, Real sqrt_temp) {
    const Real a = a_const / sqrt_temp;
    const Real z = x * x;

    // far-wing asymptotic expansion
    if (z >= Real(25)) {
        Real H = z - Real(5.5);
        H = z - Real(3.5) - Real(5) / H;
        H = z - Real(1.5) - Real(1.5) / H;
        return (a / sqrt_pi) / H;
    }

    const Real ez = exp(-z);

    // intermediate regime
    if (z > Real(3)) {
        Real H = z - B8;
        H = z - B6 + B7 / H;
        H = z + B4 + B5 / H;
        H = z - B2 + B3 / H;
        H = B0 + B1 / H;
        return ez + a * H;
    }

    // core region
    {
        Real H = z - A6;
        H = z - A4 + A5 / H;
        H = z - A2 + A3 / H;
        H = A0 + A1 / H;
        return ez * (Real(1) - a * H);
    }
}

// small helper struct for next function
struct Boundary {
    Real t;
    int axis, sign;
};

// compute_next_boundary() : identify distance to cell boundary and return
// identify which axis will be crossed at boundary as well
inline Boundary compute_next_boundary(Photon& phot, const Grid& g, 
                                      const int ix, const int iy, const int iz) {
    constexpr Real eps = (Real)1e-10;

    const Real px = phot.pos_x;
    const Real py = phot.pos_y;
    const Real pz = phot.pos_z;

    const Real dx = phot.dir_x;
    const Real dy = phot.dir_y;
    const Real dz = phot.dir_z;

    // figure out direction signs (ends up +1 or -1)
    const int sx = (dx > 0) - (dx < 0);
    const int sy = (dy > 0) - (dy < 0);
    const int sz = (dz > 0) - (dz < 0);

    // index of voxel faces that photon points to
    const int ix_face = ix + (sx > 0);
    const int iy_face = iy + (sy > 0);
    const int iz_face = iz + (sz > 0);

    // distances to next axis boundary along each respective axis
    const Real tx = (fabs(dx) > eps) ? (g.x_edges[ix_face] - px) / dx : INF;
    const Real ty = (fabs(dy) > eps) ? (g.y_edges[iy_face] - py) / dy : INF;
    const Real tz = (fabs(dz) > eps) ? (g.z_edges[iz_face] - pz) / dz : INF;

    // find minimum
    Boundary b{tx, 0, sx};
    if (ty < b.t) b = {ty, 1, sy};
    if (tz < b.t) b = {tz, 2, sz};
    return b;
}

// propogate(): takes optical depth, raytraces photon to next event
// sets next photon position before scatter
// dt = time budget for this monte carlo step, hit_time_limit set if photon runs out of time
void propogate(const Real target_tau, Photon& phot, Grid& g, int& ix, int& iy, int& iz,
               const Real dt, bool& hit_time_limit) {
    const Real eps = g.dx*1e-14;
    Real tau = 0.0;

    // define variables outside while loop
    Real sqrt_temp, n_hi, vx, vy, vz, t, dir_dot_v, x_local, sigma_alpha, dtau, s, inv_sqrt_temp;
    Real delta_t, distance;
    Boundary b;

    int cell_idx = phot.cell_idx;

    // main raytracing loop
    while (tau < target_tau) {
        if (escaped(g, phot, ix, iy, iz)) return;

        // get local cell properties
        vx   = g.vx[cell_idx];
        vy   = g.vy[cell_idx];
        vz   = g.vz[cell_idx];
        n_hi = g.hi[cell_idx];
        sqrt_temp = g.sqrt_temp[cell_idx];
        inv_sqrt_temp = Real(1.0) / sqrt_temp;

        // distance to next cell + dir
        b = compute_next_boundary(phot, g, ix, iy, iz);
        t = b.t;

        // set to min step size
        if (t < eps) t = eps;

        if (n_hi < 1e-30) {
            distance = t + eps;
            delta_t = distance * inv_c;

            // check time limit before moving
            if (phot.time + delta_t > dt) {
                // move partially to use up remaining time
                Real remaining_time = dt - phot.time;
                Real d_partial = remaining_time * c;

                phot.pos_x += d_partial * phot.dir_x;
                phot.pos_y += d_partial * phot.dir_y;
                phot.pos_z += d_partial * phot.dir_z;

                phot.time = dt; hit_time_limit = true;
                return;
            }

            phot.pos_x += distance * phot.dir_x;
            phot.pos_y += distance * phot.dir_y;
            phot.pos_z += distance * phot.dir_z;
            phot.time += delta_t;

            if (b.axis == 0) ix += b.sign;
            else if (b.axis == 1) iy += b.sign;
            else iz += b.sign;
            cell_idx = g.ny * g.nz * ix + g.nz * iy + iz;
            phot.cell_idx = cell_idx;
            continue;
        }

        // get change in x due to local temp
        phot.x *= (phot.local_sqrt_temp * inv_sqrt_temp);
        phot.local_sqrt_temp = sqrt_temp;

        // find local x due to bulk v
        dir_dot_v = vx*phot.dir_x + vy*phot.dir_y + vz*phot.dir_z;
        x_local   = phot.x - (dir_dot_v * inv_sqrt_temp) / (vth_const);
        sigma_alpha = 5.898e-12 * inv_sqrt_temp * voigt(x_local, sqrt_temp);

        // take a step in optical depth
        dtau = n_hi * sigma_alpha * (t + eps);

        // is target optical depth in current cell?
        if (tau + dtau > target_tau) {
            s = (target_tau - tau) / (n_hi * sigma_alpha);
            delta_t = s * inv_c;

            // check time limit before moving to scatter location
            if (phot.time + delta_t > dt) {
                // move partially to use up remaining time
                Real remaining_time = dt - phot.time;
                Real d_partial = remaining_time * c;
                phot.pos_x += d_partial * phot.dir_x;
                phot.pos_y += d_partial * phot.dir_y;
                phot.pos_z += d_partial * phot.dir_z;

                phot.time = dt; hit_time_limit = true;
                return;
            }

            // update pos
            phot.pos_x += s * phot.dir_x;
            phot.pos_y += s * phot.dir_y;
            phot.pos_z += s * phot.dir_z;
            phot.time += delta_t;
            return;
        }

        // move to next cell
        tau += dtau;
        distance = t + eps;
        delta_t = distance * inv_c;

        // check time limit before moving to next cell
        if (phot.time + delta_t > dt) {
            // move partially to use up remaining time
            Real remaining_time = dt - phot.time;
            Real d_partial = remaining_time * c;

            phot.pos_x += d_partial * phot.dir_x;
            phot.pos_y += d_partial * phot.dir_y;
            phot.pos_z += d_partial * phot.dir_z;

            phot.time = dt; hit_time_limit = true;
            return;
        }

        phot.pos_x += distance * phot.dir_x;
        phot.pos_y += distance * phot.dir_y;
        phot.pos_z += distance * phot.dir_z;
        phot.time += delta_t;

        if (b.axis == 0) ix += b.sign;
        else if (b.axis == 1) iy += b.sign;
        else iz += b.sign;
        cell_idx = g.ny * g.nz * ix + g.nz * iy + iz;
        phot.cell_idx = cell_idx;
    }
}

// u_parallel(): samples parallel atom velocity component
// uses rejection method following RASCAS
Real u_parallel(Real x, Real sqrt_temp, xso::rng& rng) {
    Real sign = (x >= 0.0) ? 1 : -1;
    Real xabs = fabs(x);

    // x > 8 regime: gaussian approximation using box-muller
    if (x >= 8.0) {
        Real u1 = urand(rng);
        Real u2 = urand(rng);
        if (u1 < 1e-16) u1 = 1e-16;

        Real z = sqrt(-fast_log(u1)) * cos(two_pi * u2);
        return sign * ((1.0 / xabs) + z);
    }
    // rejection method otherwise
    {
        Real a    = a_const / sqrt_temp;
        Real zeta = fast_log(a) * inv_ln_10;

        Real u0 = 0;
        if (xabs >= 0.6) {
            const Real x  = xabs;
            const Real x2 = x * x;
            const Real x4 = x2 * x2;
            const Real z2 = zeta * zeta;

            // rewritten using estrin's scheme to help parallelize
            const Real p0 =
                (2.648963 + 2.014446*zeta + 0.351479*z2)
                + x*(-4.058673 - 3.675859*zeta - 0.640003*z2);

            const Real p1 =
                (3.017395 + 2.117133*zeta + 0.370294*z2)
                + x*(-0.869789 - 0.565886*zeta - 0.096312*z2);

            const Real p2 =
                (0.110987 + 0.070103*zeta + 0.011557*z2)
                + x*(-0.005200 - 0.003240*zeta - 0.000519*z2);

            u0 = p0 + p1 * x2 + p2 * x4;
        }

        Real theta0 = atan((u0 - xabs) / a);
        Real u02 = u0*u0;
        Real eu0 = fast_exp(-u02);
        Real p   = (theta0 + pi/2) / ((1 - eu0)*theta0 + (1 + eu0)*pi/2);

        Real R1, theta, u, R2, r;
        while (true) {
            // draw two uniform randoms
            R1 = urand(rng);
            r  = urand(rng);

            // pick sampling regime
            if (R1 <= p) theta = -pi/2 + r * (theta0 + pi/2);
            else         theta = theta0 + r * (pi/2 - theta0);

            u = a * tan(theta) + xabs;
            if (!isfinite(u)) continue;

            R2 = urand(rng);

            if ((R1 <= p) && (R2 <= fast_exp(-u*u))) return u * sign;
            if ((R1 > p) && (R2 <= fast_exp(u02 - u*u))) return u * sign;
        }
    }
}

// scatter_angle(): sample scattering angle cosine
// returns mu = cos(theta) from RASCAS dipole distribution
Real scatter_angle(Real x, xso::rng& rng) {
    Real r = urand(rng);
    Real A, B;

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

// scatter(): perform resonant scattering in HI atom rest frame
void scatter(Photon& phot, Grid& g, vector<Real>& mom_x, vector<Real>& mom_y, vector<Real>& mom_z, 
             xso::rng& rng, const bool recoil, const bool isotropic) {
    int cell_idx = phot.cell_idx;

    Real sqrt_temp = g.sqrt_temp[cell_idx];
    Real vth = vth_const * sqrt_temp;

    Real u_bulk_x = g.vx[cell_idx] / vth;
    Real u_bulk_y = g.vy[cell_idx] / vth;
    Real u_bulk_z = g.vz[cell_idx] / vth;

    // transform photon freq to local bulk frame
    Real dx = phot.dir_x;
    Real dy = phot.dir_y;
    Real dz = phot.dir_z;

    Real x = phot.x;

    Real dir_dot_v = u_bulk_x * dx + u_bulk_y * dy + u_bulk_z * dz;
    Real x_local   = x - dir_dot_v;

    // sample velocity of scattering atom
    Real u_paral = u_parallel(x_local, sqrt_temp, rng);

    // transform to atom frame
    x_local -= u_paral;

    // sample perpendicular velocity components of atom using box muller
    Real u1 = urand(rng);
    Real u2 = urand(rng);
    if (u1 < 1e-16) u1 = 1e-16;

    Real R = sqrt(-log(u1));
    Real theta = two_pi * u2;

    Real u_perp1 = R * cos(theta);
    Real u_perp2 = R * sin(theta);

    // find basis that velocity components of atom are in
    // a = random vector to cross with photon direction vector
    Real ax = (fabs(dx) < 0.9) ? 1.0 : 0.0;
    Real ay = (ax != 0.0) ? 0.0 : 1.0;

    // basis vector 2, since photon's dir is basis vector 1
    Real e2x = (-dz*ay);
    Real e2y = (dz*ax);
    Real e2z = (dx*ay) - (dy*ax);

    // normalize
    Real e2_norm = 1.0 / sqrt(e2x*e2x + e2y*e2y + e2z*e2z);
    e2x *= e2_norm; e2y *= e2_norm, e2z *= e2_norm;

    // basis vector 3
    Real e3x = (dy*e2z) - (dz*e2y);
    Real e3y = (dz*e2x) - (dx*e2z);
    Real e3z = (dx*e2y) - (dy*e2x);

    // normalize if needed
    Real e3_norm_sq = e3x*e3x + e3y*e3y + e3z*e3z;
    if (fabs(e3_norm_sq - 1.0) > 1e-12) {
        Real e3_norm = 1.0 / sqrt(e3_norm_sq);
        e3x *= e3_norm; e3y *= e3_norm, e3z *= e3_norm;
    }

    // sample scattering angle
    Real cosine;
    if (!isotropic) cosine = scatter_angle(x_local, rng);
    else cosine = urand(rng)*2.0 - 1.0;
    cosine = max(-1.0, min(1.0, cosine));
    Real sine = sqrt(1.0 - cosine * cosine);

    // sample azimuthal angle uniformly
    Real phi = urand(rng) * two_pi;
    Real cosphi, sinphi;
    sincos(phi, &sinphi, &cosphi);

    // generate new direction vector
    Real new_dx = cosine*dx + sine*(cosphi*e2x + sinphi*e3x);
    Real new_dy = cosine*dy + sine*(cosphi*e2y + sinphi*e3y);
    Real new_dz = cosine*dz + sine*(cosphi*e2z + sinphi*e3z);

    // normalize in case of numerical drift
    Real norm = new_dx*new_dx + new_dy*new_dy + new_dz*new_dz;
    if (fabs(norm - 1.0) > 1e-12) {
        Real norm = 1.0 / sqrt(norm);
        new_dx *= norm; new_dy *= norm; new_dz *= norm;
    }

    // dot product of (dimensionless) velocity and new direction
    Real u_dot_k = new_dx*u_bulk_x + new_dy*u_bulk_y + new_dz*u_bulk_z;
    x_local     += u_dot_k + u_paral*cosine + sine*(u_perp1*cosphi + u_perp2*sinphi);
    if (recoil) x_local += 2.6e-4 * (1e2 / sqrt_temp) * (cosine - 1);

    // photon momentum losses added to grid (approximate since shifts in x barely shift nu)
    // scale by photon weight (number of physical photons this packet represents)
    double w = phot.weight;
    Real px = hnu_by_c * (dx - new_dx);
    Real py = hnu_by_c * (dy - new_dy);
    Real pz = hnu_by_c * (dz - new_dz);

    mom_x[cell_idx] += w * px; mom_y[cell_idx] += w * py; mom_z[cell_idx] += w * pz;

    phot.dir_x = new_dx; phot.dir_y = new_dy; phot.dir_z = new_dz;
    phot.x = x_local;
}