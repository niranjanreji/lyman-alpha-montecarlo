/* mc_grid.cpp — grid construction, HI fraction table, and
 * lyman-alpha emission rate functions. builds the 3D cartesian
 * grid and populates physical fields from user-supplied functions.
 *
 * Niranjan Reji, Raman Research Institute, March 2026 */

#include <cstdio>
#include <algorithm>

#include "common.h"
#include <rt_definitions.h>

void minimum_a_tau(Grid* grid);

/* ---- HI fraction table (temperature_HI.dat) ---- */

/**
 * @brief load the HI neutral fraction table from disk.
 * @param g_tbl_temp     output vector of temperatures [K]
 * @param g_tbl_hi_frac  output vector of corresponding HI fractions
 */
void load_hi_table(std::vector<double>& g_tbl_temp, std::vector<double>& g_tbl_hi_frac) {
    if (!g_tbl_temp.empty()) return;

    FILE* f = fopen("temperature_HI.dat", "r");
    if (!f)
        throw std::runtime_error("Failed to open temperature_HI.dat");

    double t, x;
    /* fscanf reads two whitespace separated "long floats" into t, x
     * and returns the number of values successfully read */
    while (fscanf(f, "%lf %lf", &t, &x) == 2) {
        g_tbl_temp.push_back(t);
        g_tbl_hi_frac.push_back(x);
    }
    fclose(f);

    if(g_tbl_temp.empty())
        throw std::runtime_error("temperature_HI.dat is empty");
}

/**
 * @brief interpolate the HI neutral fraction at a given temperature.
 * @param T              temperature [K]
 * @param g_tbl_temp     sorted temperature table
 * @param g_tbl_hi_frac  corresponding HI fractions
 * @return neutral hydrogen fraction x_HI at temperature T
 */
double interpolate_hi_fraction(double T, const std::vector<double>& g_tbl_temp,
                               const std::vector<double>& g_tbl_hi_frac) {
    if (T <= g_tbl_temp.front()) return g_tbl_hi_frac.front();
    if (T >= g_tbl_temp.back()) return g_tbl_hi_frac.back();

    /* binary search through g_tbl_temp to identify first
     * element greater than T, and return iterator to it */
    auto it = std::lower_bound(g_tbl_temp.begin(), g_tbl_temp.end(), T);
    int j = int(it - g_tbl_temp.begin());

    /* linearly interpolate between entries to get HI fraction */
    double w = (T - g_tbl_temp[j-1]) / (g_tbl_temp[j] - g_tbl_temp[j-1]);
    return (1.0 - w) * g_tbl_hi_frac[j-1] + w * g_tbl_hi_frac[j];
}


/* ---- lyman-alpha emission rates ---- */

/**
 * @brief fraction of case-B recombinations producing lyman-alpha.
 * @param T  gas temperature [K]
 * @return dimensionless fraction
 */
double lyalpha_rate_from_recomb(double T) {
    return 0.686 - 0.106*log10(T/10000) - 0.009*pow(T/10000, -0.44);
}

/**
 * @brief case-B recombination coefficient.
 * @param T  gas temperature [K]
 * @return recombination coefficient [cm^3/s]
 */
double caseb_recomb_coeff(double T) {
    double lambda = 315614.0 / T;
    return 2.753e-14 * pow(lambda, 1.5) / pow(1.0 + pow(lambda/2.74, 0.407), 2.242);
}

/**
 * @brief collisional excitation rate coefficient for lyman-alpha.
 * @param T  gas temperature [K]
 * @return excitation rate coefficient [cm^3/s]
 */
double excitation_rate_coeff(double T) {
    return 2.41e-6/sqrt(T) * pow(T/1e4, 0.22) * exp(-10.2/(8.617333262e-5 * T));
}


/**
 * @brief allocate and initialize the 3D cartesian grid.
 * @param sources_fn  function that fills point source positions and luminosities
 * @return pointer to newly allocated Grid
 */
Grid* init_grid(SourcesFunc sources_fn) {
    Grid *grid = new Grid();

    grid->nx = NX; grid->ny = NY; grid->nz = NZ;
    grid->Lx = LX; grid->Ly = LY; grid->Lz = LZ;

    grid->dx = grid->Lx / grid->nx;
    grid->dy = grid->Ly / grid->ny;
    grid->dz = grid->Lz / grid->nz;

    grid->dv = grid->dx * grid->dy * grid->dz;

    grid->x_edges.resize(NX + 1);
    grid->y_edges.resize(NY + 1);
    grid->z_edges.resize(NZ + 1);

    for (int i = 0; i <= NX; i++) grid->x_edges[i] = -LX/2.0 + grid->dx * i;
    for (int i = 0; i <= NY; i++) grid->y_edges[i] = -LY/2.0 + grid->dy * i;
    for (int i = 0; i <= NZ; i++) grid->z_edges[i] = -LZ/2.0 + grid->dz * i;

    grid->x_centers.resize(NX);
    grid->y_centers.resize(NY);
    grid->z_centers.resize(NZ);

    for (int i = 0; i < NX; i++) grid->x_centers[i] = 0.5 * (grid->x_edges[i] + grid->x_edges[i + 1]);
    for (int i = 0; i < NY; i++) grid->y_centers[i] = 0.5 * (grid->y_edges[i] + grid->y_edges[i + 1]);
    for (int i = 0; i < NZ; i++) grid->z_centers[i] = 0.5 * (grid->z_edges[i] + grid->z_edges[i + 1]);

    size_t n_cells = (size_t)NX * NY * NZ;
    grid->n_cells = n_cells;

    grid->sqrt_temp.resize(n_cells);
    grid->nHI.resize(n_cells);
    grid->ux.resize(n_cells);
    grid->uy.resize(n_cells);
    grid->uz.resize(n_cells);
    grid->mom_x.resize(n_cells);
    grid->mom_y.resize(n_cells);
    grid->mom_z.resize(n_cells);
    grid->energy.resize(n_cells);

    #if CORE_SKIPPING == TRUE
        grid->atau.resize(n_cells);
    #endif

    fill(grid->nHI.begin(), grid->nHI.end(), 0.0);
    fill(grid->sqrt_temp.begin(), grid->sqrt_temp.end(), 0);
    fill(grid->ux.begin(), grid->ux.end(), 0.0);
    fill(grid->uy.begin(), grid->uy.end(), 0.0);
    fill(grid->uz.begin(), grid->uz.end(), 0.0);
    fill(grid->luminosity_CDF.begin(), grid->luminosity_CDF.end(), 0.0);

    /* set up point sources */
    constexpr int MAX_SOURCES = 64;
    double pos[MAX_SOURCES][3];
    double lum[MAX_SOURCES];
    int n_sources = 0;
    sources_fn(&n_sources, pos, lum);

    grid->num_point_sources = n_sources;
    grid->ps_posx.resize(n_sources);
    grid->ps_posy.resize(n_sources);
    grid->ps_posz.resize(n_sources);
    grid->ps_lum.resize(n_sources);
    grid->ps_luminosity = 0.0;

    for (int i = 0; i < n_sources; i++) {
        grid->ps_posx[i] = pos[i][0];
        grid->ps_posy[i] = pos[i][1];
        grid->ps_posz[i] = pos[i][2];
        grid->ps_lum[i]  = lum[i];
        grid->ps_luminosity += lum[i];
    }

    return grid;
}

/**
 * @brief populate grid physical fields using provided function pointers.
 *
 * calls density_fn, temperature_fn, and velocity_fn at each cell center
 * to fill nHI, sqrt_temp, and dimensionless velocity arrays. also computes
 * grid emission (recombination + collisional excitation) and builds the
 * luminosity CDF for photon emission sampling.
 *
 * @param grid            pointer to initialized Grid
 * @param density_fn      returns total hydrogen number density n_H [cm^-3]
 * @param temperature_fn  returns gas temperature [K]
 * @param velocity_fn     fills bulk velocity components [cm/s]
 */
void build_fields(Grid* grid, DensityFunc density_fn, TemperatureFunc temperature_fn,
                  VelocityFunc velocity_fn) {
    size_t n_cells = grid->n_cells;

    /* fill physical fields using provided function pointers
     * density_fn() returns total hydrogen number density n_H
     * n_HI is computed as n_H * x_HI(T) using the ionization table */
    std::vector<double> g_tbl_temp, g_tbl_hi_frac;
    #if FULLY_NEUTRAL == FALSE
        load_hi_table(g_tbl_temp, g_tbl_hi_frac);
    #endif

    std::vector<double> grid_lum(n_cells, 0.0);
    double grid_luminosity = 0.0;

    for (int iz = 0; iz < NZ; iz++) {
        for (int iy = 0; iy < NY; iy++) {
            for (int ix = 0; ix < NX; ix++) {
                size_t idx = ix * NY * NZ + iy * NZ + iz;

                double cx = grid->x_centers[ix];
                double cy = grid->y_centers[iy];
                double cz = grid->z_centers[iz];

                double n_H  = density_fn(cx, cy, cz);
                double temp = temperature_fn(cx, cy, cz);

                double vx_raw, vy_raw, vz_raw;
                velocity_fn(cx, cy, cz, &vx_raw, &vy_raw, &vz_raw);

                #if FULLY_NEUTRAL == TRUE
                    /* density_fn returns n_HI directly, gas is fully neutral */
                    double n_HI = n_H;
                    double n_e  = 0.0;
                #else
                    double x_HI = (n_H > 0.0) ? interpolate_hi_fraction(temp, g_tbl_temp, g_tbl_hi_frac) : 0.0;
                    double n_HI = n_H * x_HI;
                    double n_e  = n_H * (1.0 - x_HI);
                #endif

                double sqrt_t = sqrt(temp);
                double inv_vth = 1.0 / (vth_const * sqrt_t);

                grid->nHI[idx] = n_HI;
                grid->sqrt_temp[idx] = sqrt_t;
                grid->ux[idx] = vx_raw * inv_vth;
                grid->uy[idx] = vy_raw * inv_vth;
                grid->uz[idx] = vz_raw * inv_vth;

                /* grid emission: recombination + collisional excitation */
                if (n_e > 0.0 && temp > 0.0) {
                    grid_lum[idx] = grid->dv * n_e
                        * (n_e  * lyalpha_rate_from_recomb(temp) * caseb_recomb_coeff(temp)
                         + n_HI * excitation_rate_coeff(temp));
                    grid_luminosity += grid_lum[idx];
                }
            }
        }
    }

    /* build luminosity CDF: grid cells first, then point sources */
    int n_sources = grid->num_point_sources;
    grid->total_luminosity = grid_luminosity + grid->ps_luminosity;
    grid->luminosity_CDF.resize(n_cells + n_sources);

    if (grid->total_luminosity > 0.0) {
        double cumulative = 0.0;
        for (size_t i = 0; i < n_cells; i++) {
            cumulative += grid_lum[i] / grid->total_luminosity;
            grid->luminosity_CDF[i] = cumulative;
        }
        for (int i = 0; i < n_sources; i++) {
            cumulative += grid->ps_lum[i] / grid->total_luminosity;
            grid->luminosity_CDF[n_cells + i] = cumulative;
        }

        /* ensure last entry is 1 */
        grid->luminosity_CDF[n_cells + n_sources - 1] = 1.0;
    }

    #if CORE_SKIPPING == TRUE 
        minimum_a_tau(grid);
    #endif
}


/* ---- core skipping computation (min(a*tau)) ---- */

/**
 * @brief precompute the non-local core-skipping estimate for every cell.
 *
 * For each cell, shoots rays in uniformly sampled directions from the cell
 * center and integrates a*k0*dl (Voigt parameter times line-center opacity
 * times path length) through successive cells. The minimum integrated a*tau
 * over all directions is stored in grid->atau[cell].
 *
 * Rays are terminated early when they exit the domain, or when a
 * discontinuity in density (>10x ratio in a*k0) or velocity (>10 vth jump)
 * is encountered. If the immediate neighbour already triggers a
 * discontinuity, that direction contributes 0 (no reliable non-local
 * estimate).
 *
 * @param grid  pointer to initialized Grid with populated physical fields
 */
/* Tolerances for discontinuity checks (following COLT) */
static constexpr double atau_density_tol  = 10.0;  /* max relative a*k0 ratio */
static constexpr double atau_velocity_tol = 10.0;  /* max velocity jump [vth]  */
static constexpr int    n_atau_dirs       = 100;    /* number of ray directions */

void minimum_a_tau(Grid* grid) {
    /* sample directions uniformly on the sphere */
    xso::rng rng;
    std::vector<double> dir_x(n_atau_dirs), dir_y(n_atau_dirs), dir_z(n_atau_dirs);

    for (int i = 0; i < n_atau_dirs; ++i) {
        double u1 = urand(rng);
        double u2 = urand(rng);

        double cosine = 2.0*u1 - 1.0;
        double phi    = two_pi*u2;
        double sine   = sqrt(1.0 - cosine*cosine);

        dir_z[i] = cosine;
        dir_y[i] = sine * std::sin(phi);
        dir_x[i] = sine * std::cos(phi);
    }

    /* precompute a*k0 per cell: a * n_HI * sigma_alpha(line center) */
    const size_t n_cells = (size_t)NX * NY * NZ;
    std::vector<double> ak0(n_cells);
    for (size_t i = 0; i < n_cells; ++i) {
        double inv_sqrt_temp = 1.0 / grid->sqrt_temp[i];
        double a = a_const * inv_sqrt_temp;
        ak0[i] = a * grid->nHI[i] * 5.898e-12 * inv_sqrt_temp * voigt_humlicek(0, a);
    }

    /* now test a*tau and find the minimum for each cell */
    for (size_t cell = 0; cell < n_cells; ++cell) {
        int iz = cell % NZ;
        int iy = (cell / NZ) % NY;
        int ix = (cell) / (NZ * NY);

        /* home cell properties for discontinuity checks */
        double ak0_home = ak0[cell];
        double vpar_home_x = grid->ux[cell];
        double vpar_home_y = grid->uy[cell];
        double vpar_home_z = grid->uz[cell];

        double atau_min = INF;

        for (int dir = 0; dir < n_atau_dirs; ++dir) {
            /* e_hat is our direction unit vector */
            double ex = dir_x[dir];
            double ey = dir_y[dir];
            double ez = dir_z[dir];

            /* parallel velocity of home cell along this direction */
            double vpar_home = vpar_home_x*ex + vpar_home_y*ey + vpar_home_z*ez;

            /* current traversal indices (mutable) */
            int cx = ix, cy = iy, cz = iz;

            /* current position */
            double px = grid->x_centers[ix];
            double py = grid->y_centers[iy];
            double pz = grid->z_centers[iz];

            /* figure out direction signs (+1 or -1) */
            const int sx = (ex > 0) ? 1 : -1;
            const int sy = (ey > 0) ? 1 : -1;
            const int sz = (ez > 0) ? 1 : -1;

            double atau_local = 0.0;
            bool first_step = true;

            /* propagation + accumulation loop */
            while (true) {
                /* face indices that e_hat points to from current cell */
                int fx = cx + (sx > 0);
                int fy = cy + (sy > 0);
                int fz = cz + (sz > 0);

                /* distances to next face along each axis */
                const double tx = (fabs(ex) > 1e-30) ? (grid->x_edges[fx] - px) / ex : INF;
                const double ty = (fabs(ey) > 1e-30) ? (grid->y_edges[fy] - py) / ey : INF;
                const double tz = (fabs(ez) > 1e-30) ? (grid->z_edges[fz] - pz) / ez : INF;

                /* find minimum distance and step to the next cell */
                double dl;
                if (tx <= ty && tx <= tz) {
                    dl = tx;
                    px += ex*dl; py += ey*dl; pz += ez*dl;
                    cx += sx;
                } else if (ty <= tz) {
                    dl = ty;
                    px += ex*dl; py += ey*dl; pz += ez*dl;
                    cy += sy;
                } else {
                    dl = tz;
                    px += ex*dl; py += ey*dl; pz += ez*dl;
                    cz += sz;
                }

                /* check if ray has exited the domain */
                if (cx < 0 || cx >= NX || cy < 0 || cy >= NY || cz < 0 || cz >= NZ) {
                    if (first_step) { atau_local = 0.0; }
                    break;
                }

                size_t next_cell = (size_t)cx * NY * NZ + (size_t)cy * NZ + cz;
                double ak0_next = ak0[next_cell];

                /* discontinuity check: density ratio */
                double ak0_ratio = (ak0_home > ak0_next)
                    ? ak0_home / ak0_next : ak0_next / ak0_home;
                if (ak0_next > 0.0 && ak0_ratio > atau_density_tol) {
                    if (first_step) { atau_local = 0.0; }
                    break;
                }

                /* discontinuity check: velocity jump along ray direction */
                double vpar_next = grid->ux[next_cell]*ex
                                 + grid->uy[next_cell]*ey
                                 + grid->uz[next_cell]*ez;
                if (fabs(vpar_home - vpar_next) > atau_velocity_tol) {
                    if (first_step) { atau_local = 0.0; }
                    break;
                }

                /* skip the home cell (don't accumulate its contribution) */
                if (first_step) {
                    first_step = false;
                    continue;
                }

                /* accumulate a*tau along the ray */
                atau_local += ak0_next * dl;

                /* early termination: can't beat current minimum */
                if (atau_local >= atau_min) break;
            }
            if (atau_local < atau_min) atau_min = atau_local;
        }
        grid->atau[cell] = atau_min;
    }
}