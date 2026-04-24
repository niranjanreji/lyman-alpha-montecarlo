/* common.h — shared constants, types, and declarations for the
 * lyman-alpha monte carlo radiative transfer code.
 *
 * Niranjan Reji, Raman Research Institute, March 2026 */

#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <limits>
#include <string>
#include <random>
#include <complex>
#include <iostream>
#include "xoshiro.h"
#include <type_traits>

/* ========== physical constants ========== */

inline constexpr double pi        = M_PI;
inline constexpr double k         = 1.380649e-16;
inline constexpr double c         = 29979245800.0;
inline constexpr double m_p       = 1.67262192595e-24;
inline constexpr double h         = 6.6260689633e-27;
inline constexpr double A_alpha   = 6.265e8;
inline constexpr double nu_alpha  = 2.466e15;
inline constexpr double vth_const = 12848.65731940276;
inline constexpr double a_const   = 0.04717165304481584;
inline constexpr double h_by_c    = 2.21021869846372176e-37;

/* voigt approximation constants — Smith et al (2015) */

inline constexpr double A0 = 15.75328153963877;
inline constexpr double A1 = 286.9341762324778;
inline constexpr double A2 = 19.05706700907019;
inline constexpr double A3 = 28.22644017233441;
inline constexpr double A4 = 9.526399802414186;
inline constexpr double A5 = 35.29217026286130;
inline constexpr double A6 = 0.8681020834678775;

inline constexpr double B0 = 0.0003300469163682737;
inline constexpr double B1 = 0.5403095364583999;
inline constexpr double B2 = 2.676724102580895;
inline constexpr double B3 = 12.82026082606220;
inline constexpr double B4 = 3.21166435627278;
inline constexpr double B5 = 32.032981933420;
inline constexpr double B6 = 9.0328158696;
inline constexpr double B7 = 23.7489999060;
inline constexpr double B8 = 1.82106170570;

/* other constants — mu_const from RASCAS dipole distribution */

inline constexpr double mu_const    = 35.97297297297297297;
inline constexpr double inv_sqrt_2  = 0.70710678118654746;
inline constexpr double sqrt_2      = 1.41421356237309515;
inline constexpr double two_pi      = 6.28318530717958623;
inline constexpr double sqrt_pi     = 1.77245385090551588;
inline constexpr double inv_sqrt_pi = 0.56418958354775629;
inline constexpr double inv_ln_10   = 0.43429448190325176;
inline constexpr double inv_c       = 0.00000000003335641;
inline constexpr double INF = std::numeric_limits<double>::infinity();


/* generate uniform random double in [0, 1) */
inline double urand(xso::rng& gen) {
    return (gen() >> 11) * 0x1.0p-53;
}


/* ========== grid information ========== */

struct Grid {

    size_t n_cells;                                  /* number of cells */

    int nx, ny, nz;                                  /* grid dimensions */
    double dx, dy, dz;                               /* grid spacing */
    double dv;                                       /* cell volume */
    double Lx, Ly, Lz;                               /* domain bounds */

    std::vector<double> x_edges, y_edges, z_edges;   /* cell edges */
    std::vector<double> x_centers, y_centers, z_centers; /* cell centers */

    /* physical fields */
    std::vector<double> sqrt_temp;
    std::vector<double> nHI;
    std::vector<double> ux;                          /* dimensionless velocity (v / vth) */
    std::vector<double> uy;
    std::vector<double> uz;
    std::vector<double> luminosity_CDF;

    /* field used for core skipping */
    std::vector<double> atau;

    int num_point_sources;                           /* number of point sources */

    /* point source positions and per-source luminosities */
    std::vector<double> ps_posx, ps_posy, ps_posz;
    std::vector<double> ps_lum;

    double ps_luminosity;                            /* total luminosity (photons/sec) */
    double total_luminosity;

    std::vector<double> mom_x, mom_y, mom_z;         /* momentum grid */
    std::vector<double> energy;                      /* deposited energy grid (optional) */
};


/* photon packet */
struct Photon {
    double dir_x, dir_y, dir_z;
    double pos_x, pos_y, pos_z;

    double x;                   /* dimensionless frequency */
    double local_sqrt_temp;     /* sqrt(temp) where x was set */
    double time;                /* propagation time elapsed */
    double weight;              /* photon packet weight */

    int     cell_idx;           /* index of cell photon is in */
    uint8_t escaped;            /* 1 if photon escapes sim box */

    uint8_t from_grid;          /* 1 if photon emitted from grid */
    xso::rng rng;               /* per-photon rng */

    int id;                     /* unique identifier */
};


/* active photon packet collection */
struct Photons {
    std::vector<Photon> data;

    inline void add_photon(const Photon& p) {
        data.push_back(p);
    }

    inline void remove_photon(const int idx) {
        data[idx] = data.back();
        data.pop_back();
    }
};

/* stores how many times the monte-carlo loop has run in program */
extern int call_number;

/* voigt profile approximations (voigt.cpp) */
double voigt_smith(double x, double a);
double voigt_tasitsiomi(double x, double a);
double voigt_humlicek(double x, double a);

/* function types for problem setup (std::function to allow capturing lambdas) */
#include <functional>
using DensityFunc     = std::function<double(double, double, double)>;
using TemperatureFunc = std::function<double(double, double, double)>;
using VelocityFunc    = std::function<void(double, double, double, double*, double*, double*)>;
typedef void   (*SourcesFunc)(int *n_sources, double pos[][3], double lum[]);

/* HI fraction table (mc_grid.cpp) */
void   load_hi_table(std::vector<double>& g_tbl_temp, std::vector<double>& g_tbl_hi_frac);
double interpolate_hi_fraction(double T, const std::vector<double>& g_tbl_temp,
                               const std::vector<double>& g_tbl_hi_frac);

Grid* init_grid(SourcesFunc sources_fn);
void  build_fields(Grid *grid, DensityFunc density_fn, TemperatureFunc temperature_fn,
                   VelocityFunc velocity_fn);
void  emit_photons(Photons& photons, Grid& grid, int total_num, int rank_num, double dt);
long long monte_carlo(Photons& p, Grid& g, double dt);

/* user-defined problem setup (user_setup.cpp) */
double user_density(double x, double y, double z);
double user_temperature(double x, double y, double z);
void   user_velocity(double x, double y, double z,
                     double *vx, double *vy, double *vz);
void   user_sources(int *n_sources, double pos[][3], double lum[]);
Grid*  load_user_grid();

/* propagation, scattering, boundary checks (physics.cpp) */
bool escaped(Grid& g, Photon& p, int& ix, int& iy, int& iz);
void propogate(const double target_tau, Photon& phot, Grid& g, int& ix, int& iy, int& iz,
               const double dt, bool& hit_time_limit);
void scatter(Photon& phot, Grid& g, xso::rng& rng);