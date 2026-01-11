#ifndef COMMON_H
#define COMMON_H

#define _USE_MATH_DEFINES
#include <bit>
#include <cmath>
#include <vector>
#include <limits>
#include <string>
#include <random>
#include <H5Cpp.h>
#include <iostream>
#include "xoshiro.h"
#include <type_traits>

using namespace std;

// ========== COMPILE-TIME PRECISION SELECTION ==========
// Define USE_SINGLE_PRECISION before including this header (or via -D flag)
// to use float instead of double. log approximation info included here

#ifdef USE_SINGLE_PRECISION
using Real = float;
using UInt = uint32_t;
inline constexpr int MANT_BITS = 23;
inline constexpr int EXP_BITS  = 8;
inline constexpr int EXP_BIAS  = 127;
#else
using Real = double;
using UInt = uint64_t;
inline constexpr int MANT_BITS = 52;
inline constexpr int EXP_BITS  = 11;
inline constexpr int EXP_BIAS  = 1023;
#endif

// ========== PHYSICAL CONSTANTS ==========

inline constexpr Real pi        = Real(M_PI);
inline constexpr Real k         = Real(1.380649e-16);
inline constexpr Real c         = Real(29979245800.0);
inline constexpr Real m_p       = Real(1.67262192595e-24);
inline constexpr Real h         = Real(6.6260689633e-27);
inline constexpr Real A_alpha   = Real(6.265e8);
inline constexpr Real nu_alpha  = Real(2.466e15);
inline constexpr Real vth_const = Real(12848.65731940276);
inline constexpr Real a_const   = Real(0.04717165304481584);
inline constexpr Real hnu_by_c  = Real(5.45039931041153810e-22);

// Approximation taken from Smith et al (2015)
// ========== VOIGT APPROXIMATION CONSTANTS ==========

inline constexpr Real A0 = Real(15.75328153963877);
inline constexpr Real A1 = Real(286.9341762324778);
inline constexpr Real A2 = Real(19.05706700907019);
inline constexpr Real A3 = Real(28.22644017233441);
inline constexpr Real A4 = Real(9.526399802414186);
inline constexpr Real A5 = Real(35.29217026286130);
inline constexpr Real A6 = Real(0.8681020834678775);

inline constexpr Real B0 = Real(0.0003300469163682737);
inline constexpr Real B1 = Real(0.5403095364583999);
inline constexpr Real B2 = Real(2.676724102580895);
inline constexpr Real B3 = Real(12.82026082606220);
inline constexpr Real B4 = Real(3.21166435627278);
inline constexpr Real B5 = Real(32.032981933420);
inline constexpr Real B6 = Real(9.0328158696);
inline constexpr Real B7 = Real(23.7489999060);
inline constexpr Real B8 = Real(1.82106170570);

// ========== OTHER CONSTANTS ==========

inline constexpr Real mu_const   = Real(35.97297297297297297);  // 11^3 / 37
inline constexpr Real inv_sqrt_2 = Real(0.70710678118654746);
inline constexpr Real sqrt_2     = Real(1.41421356237309515);
inline constexpr Real two_pi     = Real(6.28318530717958623);
inline constexpr Real sqrt_pi    = Real(1.77245385090551588);
inline constexpr Real ln_2       = Real(0.69314718055994529);
inline constexpr Real inv_ln_2   = Real(1.44269504088896339);
inline constexpr Real inv_ln_10  = Real(0.43429448190325176);
inline constexpr Real inv_c      = Real(0.00000000003335641);

inline constexpr Real INF = numeric_limits<Real>::infinity();

// ========== RANDOM NUMBERS, APPROXIMATIONS ==========

// generate uniform random Real in [0, 1)
inline Real urand(xso::rng& gen) {
    #ifdef USE_SINGLE_PRECISION
    return (gen() >> 40) * 0x1.0p-24f;  // upper 24 bits
    #else
    return (gen() >> 11) * 0x1.0p-53;   // upper 53 bits
#endif
}

// use THOR-style approximation (using IEEE exponent, mantissa bits)
// to approximate log quickly and speed up hot-paths
inline Real fast_log(Real x) {
    // reinterpret bits as those of unsigned int to extract IEEE bit pattern
    const UInt ix = bit_cast<UInt>(x);

    // bit shift to extract exponent bits, then mask exponent bits
    const int exp = int((ix >> MANT_BITS) & ((UInt(1) << EXP_BITS) - 1)) - EXP_BIAS;

    // sets exponent to 0, extracts mantissa bits to get float val of mantissa
    const UInt mant_bits = (ix & ((UInt(1) << MANT_BITS) - 1)) | (UInt(EXP_BIAS) << MANT_BITS);
    const Real m = bit_cast<Real>(mant_bits);

    const Real f  = m - 1;
    const Real f2 = f * f;

    const Real p0 = f - Real(0.5) * f2;
    const Real p1 = Real(1.0/3.0) * f - Real(0.25) * f2;

    const Real ln_m = p0 + p1 * f2;

    return Real(exp) * ln_2 + ln_m;
}

// a similar approximation for exponentials - using e^x = 2^(x / ln2),
// compute x/ln2 -> write as integer (n) + fractional part (f) -> e^x = 2^n 2^f
inline Real fast_exp(Real x) {
    Real y = x * inv_ln_2;

    int n = int(y);
    n -= (y < Real(n));
    Real f  = y - Real(n); 

    // approximate 2^f with polynomial
    const Real f2 = f * f;
    const Real p0 = Real(1) + ln_2 * f + Real(0.24022650695910069) * f2;
    const Real p1 = Real(0.05550410866482158) * f + Real(0.00961812910762848) * f2;

    const Real two_f = p0 + p1 * f2;

    // get 2^n from exponent bits
    const UInt bits = UInt(n + EXP_BIAS) << MANT_BITS;
    const Real two_n = bit_cast<Real>(bits);

    return two_n * two_f;
}

// ========== GRID INFORMATION ==========

struct Grid {

    // grid dimensions
    int nx, ny, nz;

    // grid spacing
    Real dx, dy, dz;

    // domain bounds
    Real Lx, Ly, Lz;

    // cell edges
    vector<Real> x_edges, y_edges, z_edges;

    // cell centers
    vector<Real> x_centers, y_centers, z_centers;

    // physical fields
    vector<uint16_t> sqrt_temp;
    vector<Real> hi;
    vector<Real> vx;
    vector<Real> vy;
    vector<Real> vz;
    vector<Real> lum_cdf;

    // point sources
    int n_sources;

    // positions (lum included in lum_cdf)
    vector<Real> ps_posx, ps_posy, ps_posz;

    // total luminosity (photons/sec) - sum of all sources
    double total_luminosity;

    // momentum grid
    vector<Real> mom_x, mom_y, mom_z;
};

// divides photon information in SoA style
struct Photon {
    Real dir_x, dir_y, dir_z;
    Real pos_x, pos_y, pos_z;

    Real x;
    Real local_sqrt_temp;
    Real time;
    double weight;
    int cell_idx;
    uint8_t escaped;

    // Per-photon RNG for reproducible random sequences across monte_carlo calls
    xso::rng rng;
};


struct Photons {

    vector<Photon> data;
    
    inline void add_photon(const Photon& p) {
        data.push_back(p);
    }

    inline void remove_photon(const int idx) {
        data[idx] = data.back();
        data.pop_back();
    }
};

// ========== FUNCTION DECLARATIONS ==========

// grid loading
Grid* load_grid(const string& path);

// photon SoA management (photons.cpp)
Photons* initialize_soa(int MAX_N = 100000);
int allocate_spot(Photons& p);
void deallocate_spot(Photons& p, int i);
void emit_photons(Photons& p, Grid& grid, int num, Real dt);

// physics (physics.cpp)
bool escaped(Grid& g, Photon& p, int ix, int iy, int iz);
void propogate(const Real target_tau, Photon& phot, Grid& g, int& ix, int& iy, int& iz,
               const Real dt, bool& hit_time_limit);
void scatter(Photon& p, Grid& g, vector<Real>& mom_x, vector<Real>& mom_y, vector<Real>& mom_z,
             xso::rng& rng, const bool recoil, const bool isotropic);
void monte_carlo(Photons& p, Grid& g, Real dt, int new_photon_count, bool recoil);

#endif