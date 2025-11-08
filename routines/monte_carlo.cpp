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
void monte_carlo(int max_photon_count, bool recoil, bool phi_symmetry) {
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

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        // thread local rngs, local momentum bins
        xso::rng rng_local(base_seed + (unsigned)tid * 9973u);
        uniform_real_distribution<double> uni(1e-12,1.0 - 1e-12);
        normal_distribution<double> n;
        vector<double> local_bins(nbins, 0.0);

        #pragma omp for schedule(dynamic)
        for (int photon_idx = 0; photon_idx < max_photon_count; ++photon_idx)
        {
            Photon phot;
            init_photon(phot, rng_local, uni, phi_symmetry);

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

                // update photon position
                tau_to_s(tau, phot);

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

                // scatter the photon and get radial momentum transfer
                double dp_r = scatter(phot, ix, iy, iz, rng_local, n, uni, recoil, phi_symmetry);

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

        // merge local momentum bins
        #pragma omp critical
        {
            for (int b = 0; b < nbins; ++b) momentum_bins[b] += local_bins[b];
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> total_elapsed = end - start;

    // print scatter statistics
    double avg_scatters = (double)total_scatters / max_photon_count;
    cout << "\n=== Performance Statistics ===" << endl;
    cout << "Total scatters: " << total_scatters << endl;
    cout << "Average scatters per photon: " << avg_scatters << endl;
    cout << "Total runtime: " << total_elapsed.count() << " s" << endl;

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