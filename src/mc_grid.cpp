/* mc_grid.cpp — grid construction, HI fraction table, and
 * lyman-alpha emission rate functions. builds the 3D cartesian
 * grid and populates physical fields from user-supplied functions.
 *
 * Niranjan Reji, Raman Research Institute, March 2026
 * assisted by Claude (Anthropic) */

#include <cstdio>
#include <algorithm>

#include "common.h"
#include <rt_definitions.h>

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
}
