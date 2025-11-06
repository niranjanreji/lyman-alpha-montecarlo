#ifndef COMMON_H
#define COMMON_H

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include "H5Cpp.h"

// ========== PHYSICAL CONSTANTS ==========
static const double pi        = M_PI;
static const double k         = 1.380649e-16;
static const double c         = 29979245800.0;
static const double h         = 6.62607015e-27;
static const double m_p       = 1.67262192595e-24;
static const double A_alpha   = 6.265e8;
static const double nu_alpha  = 2.466e15;
static const double vth_const = sqrt((2.0*k) / m_p);
static const double a_const   = (A_alpha*c) / (4.0*pi*nu_alpha) / vth_const;

// Approximation taken from Smith et al (2015)
// ========== VOIGT APPROXIMATION CONSTANTS ==========

constexpr double A0 = 15.75328153963877;
constexpr double A1 = 286.9341762324778;
constexpr double A2 = 19.05706700907019;
constexpr double A3 = 28.22644017233441;
constexpr double A4 = 9.526399802414186;
constexpr double A5 = 35.29217026286130;
constexpr double A6 = 0.8681020834678775;

constexpr double B0 = 0.0003300469163682737;
constexpr double B1 = 0.5403095364583999;
constexpr double B2 = 2.676724102580895;
constexpr double B3 = 12.82026082606220;
constexpr double B4 = 3.21166435627278;
constexpr double B5 = 32.032981933420;
constexpr double B6 = 9.0328158696;
constexpr double B7 = 23.7489999060;
constexpr double B8 = 1.82106170570;

// ========== STRUCTURES ==========

// CDF lookup table structure
struct CDFTable {
    std::vector<double> x, T, z, cdf;
    int nx, nT, nz;
    double eps;
    inline double at(int ix, int iT, int iz) const {
        return cdf[ix*nT*nz + iT*nz + iz];
    }
};

// 3D grid structure
struct Grid3D {
    // Grid dimensions
    int nx, ny, nz;

    // Grid spacing (cm)
    double dx, dy, dz;

    // Domain size (cm)
    double Lx, Ly, Lz;

    // Cell edges (1D arrays)
    std::vector<double> x_edges, y_edges, z_edges;

    // Cell centers (1D arrays)
    std::vector<double> x_centers, y_centers, z_centers;

    // Physical fields (3D arrays flattened to 1D)
    std::vector<int> T;            // Temperature [K]
    std::vector<double> HI;        // HI number density [cm^-3]
    std::vector<double> vx;        // Bulk velocity [cm/s]
    std::vector<double> vy;
    std::vector<double> vz;

    inline int temp(int ix, int iy, int iz) const {
        return T[ix*ny*nz + iy*nz + iz];
    }
    inline double hi(int ix, int iy, int iz) const {
        return HI[ix*ny*nz + iy*nz + iz];
    }
    inline double velx(int ix, int iy, int iz) const {
        return vx[ix*ny*nz + iy*nz + iz];
    }
    inline double vely(int ix, int iy, int iz) const {
        return vy[ix*ny*nz + iy*nz + iz];
    }
    inline double velz(int ix, int iy, int iz) const {
        return vz[ix*ny*nz + iy*nz + iz];
    }
};

// Photon structure
struct Photon {
    // photon direction, position
    double dir_x, dir_y, dir_z;
    double pos_x, pos_y, pos_z;

    // photon frequency
    double x;

    // location
    int curr_i, curr_j, curr_k;

    // temperature where x is valid
    int local_temp;
};

// ========== GLOBAL VARIABLES ==========

extern CDFTable g_table;
extern Grid3D g_grid;

// ========== FUNCTION DECLARATIONS ==========

// CDF table functions
void load_table(const std::string& path);
double sample_cdf(double x_abs, double T, double r);

// Grid functions
void load_grid(const std::string& path);

// Physics functions
inline double a_(int T) {
    return a_const / std::sqrt(T);
}

double voigt(double x, int T);

// get_cell_indices: fast index lookup for uniform grid
void get_cell_indices(Photon& phot, int& ix, int& iy, int& iz);

// escaped: returns whether photon has escaped from simulation box
bool escaped(Photon& phot);

double compute_t_to_boundary(Photon& phot, int ix, int iy, int iz);
void tau_to_s(double tau_target, Photon& phot);

double u_parallel(double x_local, double T_local, std::mt19937_64& rng,
                  std::normal_distribution<double>& norm,
                  std::uniform_real_distribution<double>& uni);

double scatter_mu(double x_local,
                  std::mt19937_64& rng,
                  std::uniform_real_distribution<double>& uni);

double scatter(Photon& phot, int ix, int iy, int iz, std::mt19937_64& rng,
               std::normal_distribution<double>& norm,
               std::uniform_real_distribution<double>& uni,
               bool recoil = true);

// Monte Carlo simulation
void monte_carlo(int max_photon_count = 100000, bool recoil = true);

#endif // COMMON_H
