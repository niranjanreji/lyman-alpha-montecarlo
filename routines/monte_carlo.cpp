/*
MONTE_CARLO.CPP / NIRANJAN REJI
- MAIN MONTE CARLO LOOP
*/

#include <omp.h>
#include <chrono>
#include <fstream>
#include "common.h"
#include <iostream>
using namespace std;

// monte_carlo(): takes photon count, recoil switch
// outputs outgoing spectrum, momentum profile text files
void monte_carlo(int max_photon_count, bool recoil) {
    // momentum transfer binning (generalize later?)
    const int nbins   = 200;
    const double rmin = 0.0;
    const double rmax = max({g_grid.Lx, g_grid.Ly, g_grid.Lz}) / sqrt(2);
    const double binw = (rmax - rmin) / nbins;

    // output vectors
    vector<double> momentum_bins(nbins, 0.0);
    vector<double> output_x_vals(max_photon_count);

    // progress tracking
    int completed_photons = 0;
    int progress_interval = max(1, max_photon_count / 20);  // update every 5%
    long long total_scatters = 0;

    // rng seed, runtime measurement
    const unsigned base_seed = 1234567u;
    auto start = chrono::high_resolution_clock::now();

    // timing accumulators for tau_to_s and scatter
    double total_tau_to_s_time = 0.0;
    double total_scatter_time = 0.0;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        // thread local rngs, local momentum bins
        mt19937_64 rng_local(base_seed + (unsigned)tid * 9973u);
        uniform_real_distribution<double> uni(1e-12,1.0 - 1e-12);
        normal_distribution<double> n;
        vector<double> local_bins(nbins, 0.0);

        // thread-local timing accumulators
        double local_tau_to_s_time = 0.0;
        double local_scatter_time = 0.0;

        #pragma omp for schedule(dynamic)
        for (int photon_idx = 0; photon_idx < max_photon_count; ++photon_idx)
        {
            Photon phot;
            phot.x = 0, phot.pos_x = 0, phot.pos_y = 0, phot.pos_z = 0;

            double dir_x = uni(rng_local)*2 - 1;
            double dir_y = uni(rng_local)*2 - 1;
            double dir_z = uni(rng_local)*2 - 1;

            // normalize direction vector
            double dir_mag = sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z);
            phot.dir_x = dir_x / dir_mag;
            phot.dir_y = dir_y / dir_mag;
            phot.dir_z = dir_z / dir_mag;

            // initialize local cell temperature for photon
            int ix, iy, iz;
            get_cell_indices(phot, ix, iy, iz);
            phot.local_temp = g_grid.temp(ix, iy, iz);

            int n_scatters = 0;
            while (!escaped(phot))
            {
                // draw random optical depth
                double r   = uni(rng_local);
                double tau = -log(r);

                // update photon position (with timing)
                auto t1 = chrono::high_resolution_clock::now();
                tau_to_s(tau, phot);
                auto t2 = chrono::high_resolution_clock::now();
                local_tau_to_s_time += chrono::duration<double>(t2 - t1).count();

                if (escaped(phot))
                {
                    // shift x to standard temperature scale
                    phot.x *= sqrt(phot.local_temp / 1e4);
                    break;
                }

                // find current cell indices
                int ix, iy, iz;
                get_cell_indices(phot, ix, iy, iz);

                // clamp to valid values for safety
                ix = max(0, min(ix, g_grid.nx - 1));
                iy = max(0, min(iy, g_grid.ny - 1));
                iz = max(0, min(iz, g_grid.nz - 1));

                int T_local = g_grid.temp(ix, iy, iz);

                // scatter the photon and get radial momentum transfer (with timing)
                auto t3 = chrono::high_resolution_clock::now();
                double dp_r = scatter(phot, ix, iy, iz, rng_local, n, uni, recoil);
                auto t4 = chrono::high_resolution_clock::now();
                local_scatter_time += chrono::duration<double>(t4 - t3).count();

                n_scatters++;

                // bin the momentum by radial distance
                double r_scatter = sqrt(phot.pos_x*phot.pos_x + phot.pos_y*phot.pos_y + phot.pos_z*phot.pos_z);
                int bin_idx = (int)((r_scatter - rmin) / binw);
                if (bin_idx >= 0 && bin_idx < nbins) {
                    local_bins[bin_idx] += dp_r;
                }
            }

            output_x_vals[photon_idx] = phot.x;

            // update progress counter
            #pragma omp atomic
            completed_photons++;

            #pragma omp atomic
            total_scatters += n_scatters;

            // print progress (only one thread at a time)
            if (completed_photons % progress_interval == 0) {
                #pragma omp critical
                {
                    auto now = chrono::high_resolution_clock::now();
                    double elapsed = chrono::duration<double>(now - start).count();
                    double percent = 100.0 * completed_photons / max_photon_count;
                    double rate = completed_photons / elapsed;
                    double eta = (max_photon_count - completed_photons) / rate;
                    cout << "Progress: " << completed_photons << "/" << max_photon_count
                         << " (" << (int)percent << "%) - "
                         << (int)rate << " photons/s - ETA: " << (int)eta << "s" << endl;
                }
            }
        }

        // merge local momentum bins and timing data
        #pragma omp critical
        {
            for (int b = 0; b < nbins; ++b) momentum_bins[b] += local_bins[b];
            total_tau_to_s_time += local_tau_to_s_time;
            total_scatter_time += local_scatter_time;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> total_elapsed = end - start;

    // print scatter statistics
    double avg_scatters = (double)total_scatters / max_photon_count;
    cout << "\n=== Performance Statistics ===" << endl;
    cout << "Total scatters: " << total_scatters << endl;
    cout << "Average scatters per photon: " << avg_scatters << endl;

    // print timing breakdown
    // Note: times are cumulative across all threads, so we report both total CPU time and per-thread average
    int num_threads = omp_get_max_threads();
    double avg_tau_to_s_time = total_tau_to_s_time / num_threads;
    double avg_scatter_time = total_scatter_time / num_threads;

    cout << "\n=== Timing Breakdown ===" << endl;
    cout << "Total wall-clock runtime: " << total_elapsed.count() << " s" << endl;
    cout << "Number of threads: " << num_threads << endl;
    cout << "\nPer-thread average times:" << endl;
    cout << "  Time in tau_to_s: " << avg_tau_to_s_time << " s ("
         << 100.0 * avg_tau_to_s_time / total_elapsed.count() << "%)" << endl;
    cout << "  Time in scatter: " << avg_scatter_time << " s ("
         << 100.0 * avg_scatter_time / total_elapsed.count() << "%)" << endl;
    cout << "  Other overhead: " << (total_elapsed.count() - avg_tau_to_s_time - avg_scatter_time) << " s ("
         << 100.0 * (total_elapsed.count() - avg_tau_to_s_time - avg_scatter_time) / total_elapsed.count() << "%)" << endl;
    cout << "\nTotal CPU time across all threads:" << endl;
    cout << "  tau_to_s: " << total_tau_to_s_time << " s" << endl;
    cout << "  scatter: " << total_scatter_time << " s" << endl;

    // write x values to output file to be plotted using python
    ofstream fout("spectrum.txt");
    for (double xv : output_x_vals) fout << xv << "\n";
    fout.close();
    cout << "Wrote spectrum.txt (" << output_x_vals.size() << " values)\n";

    // write momentum profile
    ofstream pfout("momentum_profile.txt");
    for (int b = 0; b < nbins; ++b)
    {
        double r_center = rmin + (b + 0.5) * binw;
        pfout << r_center << ' ' << momentum_bins[b] << '\n';
    }
    pfout.close();
    cout << "Wrote momentum_profile.txt (" << nbins << " bins)\n";
}