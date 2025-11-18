/*
MONTE_CARLO.CU / NIRANJAN REJI
- MAIN MONTE CARLO LOOP
*/

#include <fstream>
#include <iostream>
#include "common.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

using namespace std;

void monte_carlo_cuda(const Grid3D grid, int max_photon_count = 100000, int blocks, int threads, bool recoil = true) {
    // define vectors to hold output
    thrust::device_vector<double> output_x_vals(max_photon_count);
    thrust::device_vector<long long> output_scatters(max_photon_count);
    double* x_vals = thrust::raw_pointer_cast(output_x_vals.data());
    long long* scatters = thrust::raw_pointer_cast(output_scatters.data());

    monte_carlo_kernel<<<blocks, threads>>>(grid, max_photon_count, x_vals, scatters, recoil);
    cudaDeviceSynchronize();

    // compute total scatters
    long long total_scatters = thrust::reduce(output_scatters.begin(), output_scatters.end(), 0LL);
    double avg_scatters = (double)total_scatters / max_photon_count;

    cout << "\n=== Performance Statistics ===" << endl;
    cout << "Total scatters: " << total_scatters << endl;
    cout << "Average scatters per photon: " << avg_scatters << endl;

    // Copy x values from device to host
    thrust::host_vector<double> h_x_vals = output_x_vals;

    // Write x values to output file
    ofstream fout("spectrum.txt");
    for (double xv : h_x_vals) fout << xv << "\n";
    fout.close();
    cout << "Wrote spectrum.txt (" << h_x_vals.size() << " values)\n";
}


__global__ void monte_carlo_kernel(const Grid3D grid, int max_photons, double* x_vals, long long* scatters, bool recoil) {
    // thread id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= max_photons) return;
    const unsigned base_seed = 1234;

    // rng initialization
    PhiloxState rng;
    philox_init(rng, base_seed, tid);

    // photon initialization
    Photon phot;
    init_photon(phot, rng, grid);

    // initialize local cell temps
    int ix, iy, iz;
    long long n_scatters = 0;
    get_cell_indices(phot, grid, ix, iy, iz);
    phot.local_sqrt_temp = grid.sqrt_temp(ix, iy, iz);

    // pre allocate variables outside loop
    double r, tau;
    int T_local;

    while (!escaped(phot, grid)) {
        r   = philox_uniform(rng);
        tau = -log(r);
        tau_to_s(tau, phot, grid);

        if (escaped(phot, grid)) {
            // shift x to standard temperature scale
            phot.x *= phot.local_sqrt_temp/1e2;
            break;
        }

        // find current cell indices
        get_cell_indices(phot, grid, ix, iy, iz);
        // clamp to valid values for safety
        ix = min(ix, g_nx - 1);
        iy = min(iy, g_ny - 1);
        iz = min(iz, g_nz - 1);

        // scatter
        scatter(phot, grid, ix, iy, iz, rng);
        n_scatters++;
    }

    phot.x *= phot.local_sqrt_temp/1e2;
    x_vals[tid] = phot.x;
    scatters[tid] = n_scatters;
}