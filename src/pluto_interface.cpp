/* pluto_interface.cpp — interface between PLUTO's 1D spherical
 * hydrodynamics and the 3D cartesian RT grid. interpolates PLUTO's
 * radial arrays onto the RT grid, runs the monte carlo, and maps
 * momentum/energy back to PLUTO's radial bins.
 *
 * Niranjan Reji, Raman Research Institute, March 2026 */

#include <cstdio>
#include "common.h"
#include "pluto_interface.h"
#include <rt_definitions.h>

/**
 * @brief compute mean molecular weight from neutral fraction for pure hydrogen.
 * @param x_HI  neutral hydrogen fraction (clamped to [0, 1])
 * @return mu = 1 / (2 - x_HI)
 */
inline double mu_from_neutral_fraction(double x_HI) {
    if (x_HI < 0.0) x_HI = 0.0;
    if (x_HI > 1.0) x_HI = 1.0;
    return 1.0 / (2.0 - x_HI);
}

std::vector<double> g_tbl_temp;
std::vector<double> g_tbl_hi_frac;

/* PLUTO 1D radial arrays — file scope, set each RT call */
static double* s_rho = nullptr;   /* mass density [g/cm3] */
static double* s_pr  = nullptr;   /* pressure     [dyn/cm2] */
static double* s_vr  = nullptr;   /* radial vel   [cm/s] */
static double* s_r   = nullptr;   /* cell centers [cm] */

static int     s_i_lo = 0;        /* first physical shell (not ghost) */
static int     s_i_hi = 0;        /* last physical shell (not ghost) */
static double  s_r_face_in  = 0.0; /* inner face of first physical shell */
static double  s_r_face_out = 0.0; /* outer face of last physical shell */

/**
 * @brief linear interpolation in PLUTO's radial profile.
 * @param rad  radial distance [cm]
 * @param i0   output: lower bracket index
 * @param i1   output: upper bracket index
 * @param w    output: weight for i1 (1-w for i0)
 */
static inline void radial_interp(double rad, int& i0, int& i1, double &w) {
    /* binary search for first radius >= rad among physical shells */
    auto it = std::lower_bound(s_r + s_i_lo, s_r + s_i_hi + 1, rad);
    if (it == s_r + s_i_lo) {
        /* below first physical shell center: clamp */
        i0 = i1 = s_i_lo; w = 0.0;
    } else if (it == s_r + s_i_hi + 1) {
        /* beyond last physical shell center: clamp */
        i0 = i1 = s_i_hi; w = 0.0;
    } else {
        /* between two physical shells: interpolate */
        i1 = int(it - s_r);
        i0 = i1 - 1;
        double dr = s_r[i1] - s_r[i0];
        w = (dr > 0.0) ? (rad - s_r[i0]) / dr : 0.0;
    }
}

extern "C" {

    /**
     * @brief return total hydrogen number density at (x, y, z) by
     *        interpolating PLUTO's radial density profile.
     * @param x  cartesian x coordinate [cm]
     * @param y  cartesian y coordinate [cm]
     * @param z  cartesian z coordinate [cm]
     * @return n_H = rho / m_p [cm^-3], or 0 outside PLUTO domain
     */
    double LyaDensity(double x, double y, double z) {
        double rad = sqrt(x*x + y*y + z*z);
        if (rad > s_r_face_out || rad < s_r_face_in) return 0.0;

        int i0, i1; double w;
        radial_interp(rad, i0, i1, w);

        double rho_loc = (1.0 - w) * s_rho[i0] + w * s_rho[i1];
        return rho_loc / m_p;
    }

    /**
     * @brief return gas temperature at (x, y, z) by interpolating
     *        PLUTO's radial density and pressure profiles, then
     *        iteratively solving for T with the correct mean molecular weight.
     * @param x  cartesian x coordinate [cm]
     * @param y  cartesian y coordinate [cm]
     * @param z  cartesian z coordinate [cm]
     * @return temperature [K], or 0 outside PLUTO domain
     */
    double LyaTemperature(double x, double y, double z) {
        double rad = sqrt(x*x + y*y + z*z);
        if (rad > s_r_face_out || rad < s_r_face_in) return 0.0;

        int i0, i1; double w;
        radial_interp(rad, i0, i1, w);

        double rho = (1.0 - w) * s_rho[i0] + w * s_rho[i1];
        double pr  = (1.0 - w) * s_pr[i0] + w * s_pr[i1];

        if (!std::isfinite(rho) || !std::isfinite(pr) || rho < 0.0 || pr < 0.0) {
            throw std::domain_error("Negative/Non-finite densities / pressures detected");
        }
        if (rho == 0.0 || pr == 0.0) return 0.0;

        #if FULLY_NEUTRAL == TRUE
            /* mu = 1, no iteration needed */
            return (pr * m_p) / (rho * k);
        #else
            /* start from fully neutral estimate (mu = 1, upper bound on T)
             * iterate: T -> x_HI(T) -> mu(x_HI) -> T until convergence */
            double T = (pr * 1.0 * m_p) / (rho * k);
            for (int iter = 0; iter < 20; ++iter) {
                double x_HI  = interpolate_hi_fraction(T, g_tbl_temp, g_tbl_hi_frac);
                double mu    = mu_from_neutral_fraction(x_HI);
                double T_new = (pr * mu * m_p) / (rho * k);
                if (fabs(T_new - T) <= 1.0e-6 * T) return T_new;
                T = T_new;
            }
            return T;
        #endif
    }

    /**
     * @brief return bulk velocity at (x, y, z) by interpolating PLUTO's
     *        radial velocity and projecting onto cartesian components.
     * @param x   cartesian x coordinate [cm]
     * @param y   cartesian y coordinate [cm]
     * @param z   cartesian z coordinate [cm]
     * @param vx  output: x velocity component [cm/s]
     * @param vy  output: y velocity component [cm/s]
     * @param vz  output: z velocity component [cm/s]
     */
    void LyaVelocity(double x, double y, double z,
                     double *vx, double *vy, double *vz) {

        double rad = sqrt(x*x + y*y + z*z);
        if (rad > s_r_face_out || rad < s_r_face_in) {
            *vx = *vy = *vz = 0.0;
            return;
        }

        int i0, i1; double w;
        radial_interp(rad, i0, i1, w);

        double vr_loc = (1.0 - w) * s_vr[i0] + w * s_vr[i1];
        double inv_r = 1.0 / rad;
        *vx = vr_loc * x * inv_r;
        *vy = vr_loc * y * inv_r;
        *vz = vr_loc * z * inv_r;
    }

    /* the three functions below are defined for Analysis() in init.c
     * guard against being called before the HI table is loaded */

    /**
     * @brief compute temperature from density and pressure (for PLUTO diagnostics).
     * @param rho  mass density [g/cm^3]
     * @param pr   pressure [dyn/cm^2]
     * @return temperature [K], or 0 if inputs are invalid
     */
    double LyaTemperatureFromHydro(double rho, double pr) {
        if (!std::isfinite(rho) || !std::isfinite(pr) || rho <= 0.0 || pr <= 0.0) return 0.0;

        #if FULLY_NEUTRAL == TRUE
            return (pr * m_p) / (rho * k);
        #else
            if (g_tbl_temp.empty()) load_hi_table(g_tbl_temp, g_tbl_hi_frac);

            double T = (pr * 1.0 * m_p) / (rho * k);
            for (int iter = 0; iter < 20; ++iter) {
                double x_HI  = interpolate_hi_fraction(T, g_tbl_temp, g_tbl_hi_frac);
                double mu    = mu_from_neutral_fraction(x_HI);
                double T_new = (pr * mu * m_p) / (rho * k);
                if (fabs(T_new - T) <= 1.0e-6 * T) return T_new;
                T = T_new;
            }
            return T;
        #endif
    }

    /**
     * @brief return the HI neutral fraction at a given temperature.
     * @param T  temperature [K]
     * @return neutral hydrogen fraction x_HI
     */
    double LyaNeutralFractionFromTemperature(double T) {
        #if FULLY_NEUTRAL == TRUE
            return 1.0;
        #else
            if (g_tbl_temp.empty()) load_hi_table(g_tbl_temp, g_tbl_hi_frac);
            return interpolate_hi_fraction(T, g_tbl_temp, g_tbl_hi_frac);
        #endif
    }

    /**
     * @brief compute neutral hydrogen number density from density and pressure.
     * @param rho  mass density [g/cm^3]
     * @param pr   pressure [dyn/cm^2]
     * @return n_HI [cm^-3], or 0 if inputs are invalid
     */
    double LyaNeutralHydrogenNumberDensity(double rho, double pr) {
        if (!std::isfinite(rho) || !std::isfinite(pr) || rho <= 0.0 || pr <= 0.0) return 0.0;
        #if FULLY_NEUTRAL == TRUE
            return rho / m_p;
        #else
            double T = LyaTemperatureFromHydro(rho, pr);
            double n_H = rho / m_p;
            return n_H * interpolate_hi_fraction(T, g_tbl_temp, g_tbl_hi_frac);
        #endif
    }

    #if PRINT_TO_FILE == YES
        static FILE* saved_stdout = nullptr;
        static FILE* saved_stderr = nullptr;
    #endif

    /**
     * @brief main entry point called by PLUTO each hydro step.
     *
     * interpolates PLUTO's 1D radial arrays onto the 3D RT grid,
     * runs the monte carlo, maps momentum/energy back to PLUTO's
     * radial bins, and computes boundary flux diagnostics.
     *
     * @param d  pointer to LyaData struct with PLUTO arrays and output buffers
     */
    void LyaRadiativeTransfer(LyaData* d) {
        #if PRINT_TO_FILE == YES
            freopen("pluto.log", "a", stdout);
            freopen("pluto.log", "a", stderr);
        #endif

        static Grid* grid = nullptr;
        static Photons* p = nullptr;

        if (!grid) grid = init_grid(user_sources);
        if (!p)       p = new Photons();
        #if FULLY_NEUTRAL == FALSE
            load_hi_table(g_tbl_temp, g_tbl_hi_frac);
        #endif

        int n        = d->n;
        double  dt   = d->dt;
        double* rho  = d->rho;
        double* vr   = d->vr;
        double* r    = d->r;
        double* pr   = d->pr;
        double* dV   = d->dV;
        double* out_force  = d->out_force;
        double* out_energy = d->out_energy;
        double* photon_energy = d->photon_energy;
        double* photon_count  = d->photon_count;
        double* grid_photon_count = d->grid_photon_count;
        double gamma_gas = d->gamma_gas;

        /* validate PLUTO input arrays before any computation */
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(rho[i]) || !std::isfinite(pr[i]) || !std::isfinite(vr[i])) {
                std::cerr << "ERROR: NaN/Inf in PLUTO input at cell " << i << std::endl;
                std::cerr << "  rho=" << rho[i] << " pr=" << pr[i] << " vr=" << vr[i] << " r=" << r[i] << std::endl;
                throw std::domain_error("NaN/Inf detected in PLUTO input arrays");
            }
            if (rho[i] < 0.0) {
                std::cerr << "ERROR: Negative density at cell " << i << ": rho=" << rho[i] << std::endl;
                throw std::domain_error("PLUTO returned negative density");
            }
            if (pr[i] < 0.0) {
                std::cerr << "ERROR: Negative pressure at cell " << i << ": pr=" << pr[i] << std::endl;
                throw std::domain_error("PLUTO returned negative pressure");
            }
        }

        /* physical domain bounds from PLUTO's ghost cell count */
        int nghost = d->nghost;
        int i_lo = nghost;              /* = IBEG */
        int i_hi = n - 1 - nghost;     /* = IEND */
        if (i_lo >= i_hi) {
            throw std::runtime_error("No physical shells available for RT mapping.");
        }

        double r_max = 0.5 * sqrt(LX*LX + LY*LY + LZ*LZ);
        double dr_in  = r[i_lo + 1] - r[i_lo];
        double dr_out = r[i_hi] - r[i_hi - 1];
        double r_face_in  = r[i_lo] - 0.5*dr_in;
        double r_face_out = r[i_hi] + 0.5*dr_out;

        if (r_max < r_face_out) throw std::domain_error("PLUTO domain doesn't fit! Ensure your RT grid is larger");

        double area_in  = 4.0 * pi * r_face_in * r_face_in;
        double area_out = 4.0 * pi * r_face_out * r_face_out;

        /* helper to compute energy density plus pressure to compute
         * outflow/inflow of energy through mass outflow / inflow */
        auto energy_density_plus_p = [gamma_gas](double rho, double v, double pr) {
            double kinetic = 0.5 * rho * v * v;
            double thermal = pr / (gamma_gas - 1.0);
            return kinetic + thermal + pr;
        };

        /* compute boundary mass, energy, momentum flux due to matter
         * not expecting inflows of matter, so check for that
         * to be clear, in = inner boundary, out = outer boundary */
        double mdot_in = (vr[i_lo] < 0.0) ? area_in * rho[i_lo] * vr[i_lo] : 0.0;
        double mdot_out = (vr[i_hi] > 0.0) ? area_out * rho[i_hi] * vr[i_hi] : 0.0;
        double pdot_in = (vr[i_lo] < 0.0) ? area_in * rho[i_lo] * vr[i_lo] * vr[i_lo] : 0.0;
        double pdot_out = (vr[i_hi] > 0.0) ? area_out * rho[i_hi] * vr[i_hi] * vr[i_hi] : 0.0;
        double edot_in = (vr[i_lo] < 0.0) ? area_in * energy_density_plus_p(rho[i_lo], vr[i_lo], pr[i_lo]) * vr[i_lo] : 0.0;
        double edot_out = (vr[i_hi] > 0.0) ? area_out * energy_density_plus_p(rho[i_hi], vr[i_hi], pr[i_hi]) * vr[i_hi] : 0.0;

        g_lya_cum_mass_flux += (mdot_in - mdot_out) * dt;
        g_lya_cum_momentum_flux += (pdot_in - pdot_out) * dt;
        g_lya_cum_energy_flux += (edot_in - edot_out) * dt;

        s_rho = rho; s_pr = pr; s_vr = vr; s_r = r;
        s_i_lo = i_lo; s_i_hi = i_hi;
        s_r_face_in  = r_face_in;
        s_r_face_out = r_face_out;

        int ncells = (int)NX*NY*NZ;

        /* interpolate data from PLUTO to RT grid */
        build_fields(grid, LyaDensity, LyaTemperature, LyaVelocity);

        /* run monte carlo */
        d->num_scatters = monte_carlo(*p, *grid, dt);

        /* collect photon count and energy diagnostics */
        std::fill(photon_count, photon_count + n, 0.0);
        std::fill(grid_photon_count, grid_photon_count + n, 0.0);
        std::fill(photon_energy, photon_energy + n, 0.0);

        for (size_t photon_idx = 0; photon_idx < p->data.size(); ++photon_idx) {
            const Photon& phot = p->data[photon_idx];

            /* calculate radial distance for binning */
            double rad = sqrt(phot.pos_x*phot.pos_x + phot.pos_y*phot.pos_y + phot.pos_z*phot.pos_z);
            rad = std::min(rad, r[i_hi]);

            /* find bin index using binary search */
            auto it = std::lower_bound(r + i_lo, r + i_hi + 1, rad);
            int i_bin;

            if (it == r + i_lo) i_bin = i_lo;
            else if (it == r + i_hi + 1) i_bin = i_hi;
            else i_bin = int(it - r);

            photon_count[i_bin] += 1.0;
            if (phot.from_grid) grid_photon_count[i_bin] += 1.0;
            double nu_packet = nu_alpha * (1.0 + phot.x*phot.local_sqrt_temp*vth_const/c);
            photon_energy[i_bin] += phot.weight * h * nu_packet;
        }

        /* map 3D momentum, energy back to PLUTO 1D radial bins */
        std::fill(out_force, out_force + n, 0.0);
        std::fill(out_energy, out_energy + n, 0.0);

        /* vector to store how many RT cells map to
         * each PLUTO radial bin */
        static std::vector<int> num_cells;
        num_cells.resize(n);
        std::fill(num_cells.begin(), num_cells.end(), 0);

        for (int idx = 0; idx < ncells; ++idx) {
            int iz = idx % grid->nz;
            int iy = (idx / grid->nz) % grid->ny;
            int ix = idx / (grid->ny * grid->nz);

            double x = grid->x_centers[ix];
            double y = grid->y_centers[iy];
            double z = grid->z_centers[iz];
            double rad = sqrt(x*x + y*y + z*z);

            if (rad >= r_face_in && rad <= r_face_out) {
                auto it = std::lower_bound(r + i_lo, r + i_hi + 1, rad);
                int pluto_idx;
                if (it == r + i_lo) pluto_idx = i_lo;
                else if (it == r + i_hi + 1) pluto_idx = i_hi;
                else {
                    int i1 = int(it - r);
                    int i0 = i1 - 1;
                    pluto_idx = (fabs(rad - r[i0]) <= fabs(r[i1] - rad)) ? i0 : i1;
                }

                /* sum momentum densities, energy densities */
                double mom_radial = (grid->mom_x[idx]*x + grid->mom_y[idx]*y + grid->mom_z[idx]*z) / (rad * grid->dv);
                out_force[pluto_idx] += mom_radial;
                num_cells[pluto_idx] += 1;

                #if ENERGY_DEPOSIT == DIRECT
                    double energy_cell = (grid->energy[idx]) / (grid->dv);
                    out_energy[pluto_idx] += energy_cell;
                #else
                    out_energy[pluto_idx] = 0;
                #endif
            }
        }

        /* average accumulated mom/energy densities */
        for (int i = 0; i < n; ++i) {
            if (num_cells[i] > 0) {
                out_force[i] = out_force[i] / (num_cells[i] * dt);
                out_energy[i] = out_energy[i] / (num_cells[i] * dt);
            }
            else out_force[i] = 0.0;
        }

        /*
         *
         * TO BUILD: SMOOTHING OPTION FOR FORCE, ENERGY
         *
         *
        */

        /* check for NaN/Inf in output */
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(out_force[i]) || !std::isfinite(out_energy[i])) {
                std::cerr << "ERROR: out_force[" << i << "] = " << out_force[i]
                          << ", out_energy[" << i << "] = " << out_energy[i]
                          << " (NaN/Inf detected)" << std::endl;
                std::cerr << "  dt = " << dt << std::endl;
                throw std::runtime_error("NaN detected in RT output force");
            }
        }

        /* clear grid momentum, energy for next timestep */
        std::fill(grid->mom_x.begin(), grid->mom_x.end(), 0.0);
        std::fill(grid->mom_y.begin(), grid->mom_y.end(), 0.0);
        std::fill(grid->mom_z.begin(), grid->mom_z.end(), 0.0);
        std::fill(grid->energy.begin(), grid->energy.end(), 0.0);

        #if PRINT_TO_FILE == YES
            freopen("/dev/tty", "w", stdout);
            freopen("/dev/tty", "w", stderr);
        #endif
    }
}
