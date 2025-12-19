// ---------------------------------------
// monte_carlo.cpp - main monte carlo + hydro loop
// ---------------------------------------

#include <omp.h>
#include <chrono>
#include <fstream>
#include "common.h"

// monte_carlo(): the main monte carlo loop
// evolves all photons for duration dt, emits new photons, deposits momentum to grid
// photons that don't escape within dt are preserved for the next call
void monte_carlo(Photons& p, Grid& g, Real dt, int new_photon_count, bool recoil) {
    // momentum grids for each thread
    int n_cells = g.nx * g.ny * g.nz;
    int nthreads = omp_get_max_threads();

    vector<vector<Real>> mom_x_thread(nthreads, vector<Real>(n_cells, 0.0));
    vector<vector<Real>> mom_y_thread(nthreads, vector<Real>(n_cells, 0.0));
    vector<vector<Real>> mom_z_thread(nthreads, vector<Real>(n_cells, 0.0));

    // output spectrum (escaped photon frequencies)
    vector<Real> output_x_vals;
    output_x_vals.reserve(new_photon_count);

    // progress tracking
    int completed_photons = 0;
    long long total_scatters = 0;

    const unsigned base_seed = 0;
    xso::rng rng_main(base_seed);

    // reset time for all carried-over photons
    #pragma omp parallel for
    for (size_t i = 0; i < p.data.size(); ++i) {
        p.data[i].time = 0;
    }

    // emit new photons (single-threaded since allocation modifies shared state)
    emit_photons(p, g, rng_main, new_photon_count, dt);

    // capture size after emission
    size_t n_active = p.data.size();
    int progress_interval = max(1, (int)n_active / 20);  // update every 5%

    auto start = chrono::high_resolution_clock::now();

    // main evolution loop over ALL active photons (new + carried over)
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        xso::rng rng_local(base_seed + (unsigned)tid * 9973u);

        #pragma omp for schedule(dynamic)
        for (size_t photon_idx = 0; photon_idx < n_active; ++photon_idx)
        {
            Photon& phot = p.data[photon_idx];

            int iz = phot.cell_idx % g.nz;
            int iy = (phot.cell_idx / g.nz) % g.ny;
            int ix = phot.cell_idx / (g.ny * g.nz);

            bool hit_time_limit = false;
            Real r, tau;
            int n_scatters = 0;

            // evolve until escaped or time limit reached
            while (!escaped(g, phot, ix, iy, iz) && !hit_time_limit)
            {
                r   = urand(rng_local);
                tau = -fast_log(r);

                // propogate photon through tau (may hit time limit)
                propogate(tau, phot, g, ix, iy, iz, dt, hit_time_limit);

                // check for escape or time limit after propogation
                if (escaped(g, phot, ix, iy, iz) || hit_time_limit) break;

                // if no escape and still have time, scatter
                scatter(phot, g, mom_x_thread[tid], mom_y_thread[tid], mom_z_thread[tid],
                        rng_local, recoil, false);
                n_scatters++;
            }

            // if escaped, shift frequency to standard temperature scale for output
            if (phot.escaped) phot.x *= phot.local_sqrt_temp / 1e2;

            // if hit_time_limit, photon stays active for next call (no action needed)

            // update progress counters
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
                    double percent = 100.0 * completed_photons / n_active;
                    double rate = completed_photons / elapsed;
                    double eta = (n_active - completed_photons) / rate;
                    cout << "Progress: " << completed_photons << "/" << n_active
                         << " (" << (int)percent << "%) - "
                         << (int)rate << " photons/s - ETA: " << (int)eta << "s" << endl;
                }
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> total_elapsed = end - start;

    // collect escaped frequencies and remove escaped photons
    for (Photon& phot : p.data) {
        if (phot.escaped) output_x_vals.push_back(phot.x);
    }
    p.data.erase(
        remove_if(p.data.begin(), p.data.end(), [](const Photon& phot) { return phot.escaped; }),
        p.data.end()
    );

    // accumulate thread-local momentum grids into global grid
    for (int tid = 0; tid < nthreads; ++tid) {
        for (int i = 0; i < n_cells; ++i) {
            g.mom_x[i] += mom_x_thread[tid][i];
            g.mom_y[i] += mom_y_thread[tid][i];
            g.mom_z[i] += mom_z_thread[tid][i];
        }
    }

    // print performance statistics
    double avg_scatters = (n_active > 0) ? (double)total_scatters / n_active : 0;
    cout << "\n=== Performance Statistics ===" << endl;
    cout << "Total photons processed: " << n_active << endl;
    cout << "Photons escaped: " << output_x_vals.size() << endl;
    cout << "Photons carried over: " << p.data.size() << endl;
    cout << "Total scatters: " << total_scatters << endl;
    cout << "Average scatters per photon: " << avg_scatters << endl;
    cout << "Total runtime: " << total_elapsed.count() << " s" << endl;

    // write escaped photon frequencies to output file
    if (!output_x_vals.empty()) {
        ofstream fout("output/spectrum.txt");
        for (Real xv : output_x_vals) fout << xv << "\n";
        fout.close();
        cout << "Wrote spectrum.txt (" << output_x_vals.size() << " values)\n";
    }
}
