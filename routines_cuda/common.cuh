#ifndef COMMON_CUDA_H
#define COMMON_CUDA_H

#define _USE_MATH_DEFINES
#include <cmath>
#include "H5Cpp.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

// use cuRAND's Philox implementation for RNG
typedef curandStatePhilox4_32_10_t PhiloxState;

// simple wrapper functions for RNG
__device__ __forceinline__ void philox_init(PhiloxState& state, uint64_t seed, uint64_t sequence) {
    curand_init(seed, sequence, 0, &state);
}

__device__ __forceinline__ double philox_uniform(PhiloxState& state) {
    return curand_uniform_double(&state);
}

__device__ __forceinline__ double philox_normal(PhiloxState& state) {
    return curand_normal_double(&state);
}

__device__ __forceinline__ void philox_normal_pair(PhiloxState& state, double& z0, double& z1) {
    double2 result = curand_normal2_double(&state);
    z0 = result.x;
    z1 = result.y;
}

// ========== PHYSICAL CONSTANTS (GPU Constant Memory) ==========

__constant__ double pi          = M_PI;
__constant__ double k           = 1.380649e-16;
__constant__ double c           = 29979245800.0;
__constant__ double h           = 6.62607015e-27;
__constant__ double m_p         = 1.67262192595e-24;
__constant__ double A_alpha     = 6.265e8;
__constant__ double nu_alpha    = 2.466e15;
__constant__ double vth_const   = 1.28486551932888e5;  // sqrt((2.0*k) / m_p)
__constant__ double a_const     = 4.701764810494981e-4; // (A_alpha*c) / (4.0*pi*nu_alpha) / vth_const
__constant__ double hnualphabyc = 5.445824663086854e-13; // (h * nu_alpha)/c
__constant__ double INF = __longlong_as_double(0x7ff0000000000000ULL);

// Approximation taken from Smith et al (2015)
// ========== VOIGT APPROXIMATION CONSTANTS ==========

__constant__ double A0 = 15.75328153963877;
__constant__ double A1 = 286.9341762324778;
__constant__ double A2 = 19.05706700907019;
__constant__ double A3 = 28.22644017233441;
__constant__ double A4 = 9.526399802414186;
__constant__ double A5 = 35.29217026286130;
__constant__ double A6 = 0.8681020834678775;

__constant__ double B0 = 0.0003300469163682737;
__constant__ double B1 = 0.5403095364583999;
__constant__ double B2 = 2.676724102580895;
__constant__ double B3 = 12.82026082606220;
__constant__ double B4 = 3.21166435627278;
__constant__ double B5 = 32.032981933420;
__constant__ double B6 = 9.0328158696;
__constant__ double B7 = 23.7489999060;
__constant__ double B8 = 1.82106170570;

// ========== OTHER CONSTS ========

__constant__ double inv_sqrt_2 = 0.7071067811865476; // sqrt(1.0/2.0)
__constant__ double two_pi     = 6.283185307179586;  // 2.0*pi

// ========== GRID METADATA (CONSTANT MEMORY FOR FAST ACCESS) ==========

// Grid dimensions (accessed by every thread for indexing)
__constant__ int g_nx, g_ny, g_nz;

// Grid spacing (used in cell index calculations)
__constant__ double g_dx, g_dy, g_dz;

// Domain bounds (used for boundary checks)
__constant__ double g_x_min, g_y_min, g_z_min;
__constant__ double g_x_max, g_y_max, g_z_max;

// ========== STRUCTURES ==========

// 3D grid structure (GPU version with texture memory support)
// NOTE: Dimensions/spacing are in __constant__ memory (g_nx, g_ny, g_nz, g_dx, g_dy, g_dz)
struct Grid3D {
    // 1D edge/center arrays (texture memory for spatial locality)
    cudaTextureObject_t x_edges;
    cudaTextureObject_t y_edges;
    cudaTextureObject_t z_edges;
    cudaTextureObject_t x_centers;
    cudaTextureObject_t y_centers;
    cudaTextureObject_t z_centers;

    // Physical fields (3D arrays flattened to 1D, bound to texture memory)
    cudaTextureObject_t sqrt_T;
    cudaTextureObject_t HI;
    cudaTextureObject_t vx;
    cudaTextureObject_t vy;
    cudaTextureObject_t vz;

    // Device accessor methods - use g_nx, g_ny, g_nz from constant memory
    __device__ inline int sqrt_temp(int ix, int iy, int iz) const {
        int idx = ix * g_ny * g_nz + iy * g_nz + iz;
        return tex1Dfetch<int>(sqrt_T, idx);
    }
    __device__ inline double hi(int ix, int iy, int iz) const {
        int idx = ix * g_ny * g_nz + iy * g_nz + iz;
        return tex1Dfetch<double>(HI, idx);
    }
    __device__ inline double velx(int ix, int iy, int iz) const {
        int idx = ix * g_ny * g_nz + iy * g_nz + iz;
        return tex1Dfetch<double>(vx, idx);
    }
    __device__ inline double vely(int ix, int iy, int iz) const {
        int idx = ix * g_ny * g_nz + iy * g_nz + iz;
        return tex1Dfetch<double>(vy, idx);
    }
    __device__ inline double velz(int ix, int iy, int iz) const {
        int idx = ix * g_ny * g_nz + iy * g_nz + iz;
        return tex1Dfetch<double>(vz, idx);
    }
};

// Photon structure (already GPU-compatible - POD type)
struct Photon {
    // photon direction, position
    double dir_x, dir_y, dir_z;
    double pos_x, pos_y, pos_z;

    // photon frequency
    double x;

    // location
    int curr_i, curr_j, curr_k;

    // temperature where x is valid
    int local_sqrt_temp;
};

// ========== FUNCTION DECLARATIONS ==========

// host-side grid loading function
Grid3D load_grid(const std::string& path);

// host-side grid cleanup function
void free_grid(Grid3D& grid);

// device functions (callable from kernels)
__device__ __forceinline__ double voigt(double x, int sqrt_T);

__device__ __forceinline__ void get_cell_indices(const Photon& phot, const Grid3D& grid,
                                  int& ix, int& iy, int& iz);

__device__ __forceinline__ void init_photon(Photon& phot, PhiloxState& rng_state, const Grid3D& grid);

__device__ __forceinline__ bool escaped(const Photon& phot, const Grid3D& grid);

__device__ __forceinline__ double compute_t_to_boundary(const Photon& phot, const Grid3D& grid,
                                         int ix, int iy, int iz);

__device__ void tau_to_s(const double tau_target, Photon& phot, const Grid3D& grid);

__device__ double u_parallel(const double x_local, const double sqrt_T_local, PhiloxState& rng_state);

__device__ double scatter_mu(const double x_local, PhiloxState& rng_state);

__device__ void scatter(Photon& phot, const Grid3D& grid,
                          const int ix, const int iy, const int iz, PhiloxState& rng_state,
                          const bool recoil = true);

// main Monte Carlo kernel (global kernel entry point)
__global__ void monte_carlo_kernel(Grid3D grid, unsigned long seed,
                                   int n_photons, bool recoil);

// host wrapper function for launching Monte Carlo
void monte_carlo_cuda(int max_photon_count = 100000, bool recoil = true);

#endif // COMMON_CUDA_H
