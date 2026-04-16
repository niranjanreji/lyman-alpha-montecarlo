/* monte_carlo.cpp — main monte carlo loop. emits photon packets,
 * propagates them through the grid, handles scattering, and
 * collects diagnostics. supports MPI and OpenMP parallelism.
 *
 * Niranjan Reji, Raman Research Institute, March 2026 */

#include <omp.h>
#include <chrono>
#include <fstream>
#include "common.h"
#include "pluto_interface.h"
#include <rt_definitions.h>

#ifdef PARALLEL
    #include <mpi.h>
#endif

int call_number = -1;

/* cumulative photon energy tracking (used by both standalone and coupled modes) */
double g_lya_cum_photon_energy_injected = 0.0;
double g_lya_cum_photon_energy_escaped  = 0.0;

/**
 * @brief run the monte carlo radiative transfer loop.
 *
 * emits new photon packets, propagates all active photons through
 * the grid until they escape or exhaust the time budget, deposits
 * momentum and energy, and writes optional spectrum/position output.
 * 
 * momentum deposited per cell can be written to a file if the 
 * corresponding flag is switched on, but this reduces all 
 * momentum arrays within the monte carlo loop so beware.
 *
 * @param p   photon collection (carries over between calls)
 * @param g   grid with physical fields and momentum arrays
 * @param dt  time budget for this step [s]
 * @return total number of scattering events across all ranks
 */
long long monte_carlo(Photons& p, Grid& g, double dt) {
    ++call_number;
    omp_set_num_threads(OMP_NUM_THREADS);
    auto start_time = std::chrono::high_resolution_clock::now();

    int rank = 0, nprocs = 1;
    #ifdef PARALLEL
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    #endif

    long long new_photon_count;
    if (DT_PHOTONS > 0)
        new_photon_count = std::max(N_PHOTONS * (dt / DT_PHOTONS), 1.0);
    else
        new_photon_count = N_PHOTONS;

    int base_count = new_photon_count / nprocs;
    int remainder  = new_photon_count % nprocs;
    int start      = call_number % nprocs;

    /* load distribution trick to spread photon load evenly among ranks */
    int local_count = base_count;
    #ifdef PARALLEL
        if ( (rank - start + nprocs) % nprocs < remainder ) local_count++;
    #endif

    double injected_before = g_lya_cum_photon_energy_injected;
    emit_photons(p, g, new_photon_count, local_count, dt);
    double injected_local = g_lya_cum_photon_energy_injected - injected_before;

    /* debug output */
    int n_active = static_cast<int>(p.data.size());
    #ifdef PARALLEL
        MPI_Allreduce(MPI_IN_PLACE, &n_active, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    #endif
    if (rank == 0) {
        std::cout << "[MC #" << call_number << "] dt=" << dt << "s, "
                  << n_active << " active photons" << std::endl;
    }
    int n_local = static_cast<int>(p.data.size());

    /* reset time for all active photons */
    #pragma omp parallel for
    for (int i = 0; i < n_local; ++i) p.data[i].time = 0.0;

    /* diagnostic variables */
    int completed_local = 0;
    long long scatters_local = 0;
    double escaped_energy_local = 0.0;

    /* progress tracking (shared across threads) */
    #if VERBOSE_OUTPUT == TRUE
        int progress_count = 0;
        int progress_step = std::max(1, n_active / (20 * nprocs));
    #endif

    /* position buffer to store positions during monte-carlo run */
    #if POSITION_OUTPUT == TRUE
        std::vector<std::vector<std::array<double, 4>>> pos_buffers(OMP_NUM_THREADS);
    #endif

    #pragma omp parallel for schedule(dynamic) reduction(+:completed_local,scatters_local,escaped_energy_local)
    for (int photon_idx = 0; photon_idx < n_local; ++photon_idx) {
        Photon& phot = p.data[photon_idx];

        int iz = phot.cell_idx % g.nz;
        int iy = (phot.cell_idx / g.nz) % g.ny;
        int ix = phot.cell_idx / (g.ny * g.nz);

        bool hit_time_limit = false;
        int  n_scatters = 0;

        /* main propagation loop */
        while (!escaped(g, phot, ix, iy, iz) && !hit_time_limit) {
            //std::cout << "here0" << std::endl;
            /* sample random optical depth from PDF */
            double r   = urand(phot.rng);
            double tau = -log(r);
            //std::cout << "here1" << std::endl;
            
            propogate(tau, phot, g, ix, iy, iz, dt, hit_time_limit);
            if (escaped(g, phot, ix, iy, iz) || hit_time_limit) break;

            //std::cout << "here2" << std::endl;
            scatter(phot, g, phot.rng);
            #if POSITION_OUTPUT == TRUE
                if (n_scatters % POSITION_INTERVAL == 0) {
                    int tid = omp_get_thread_num();
                    pos_buffers[tid].push_back({phot.pos_x, phot.pos_y, phot.pos_z, static_cast<double>(phot.id)});
                }
            #endif
            //std::cout << "here3" << std::endl;

            ++n_scatters;
        }
        
        if (phot.escaped) {
            double nu_escape = nu_alpha * (1.0 + phot.x * phot.local_sqrt_temp * vth_const / c);
            if (std::isfinite(nu_escape) && nu_escape > 0.0) {
                escaped_energy_local += phot.weight * h * nu_escape;
            }
            /* convert to standard temperature scale for spectrum.txt */
            phot.x *= phot.local_sqrt_temp / 1e2;
        }

        completed_local += 1;
        scatters_local += n_scatters;

        #if VERBOSE_OUTPUT == TRUE
        {
            int cur;
            #pragma omp atomic capture
            cur = ++progress_count;
            if (rank == 0 && cur % progress_step == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                double rate = (elapsed > 0.0) ? cur / elapsed : 0.0;
                #pragma omp critical
                {
                    int global_est = cur * nprocs;
                    std::cout << "  Progress: " << global_est << "/" << n_active
                              << " - " << (int)(rate * nprocs) << " photons/s" << std::endl;
                }
            }
        }
        #endif
    }

    /* local owned diagnostics */
    int escaped_local = 0;
    int carry_local = 0;
    std::vector<double> escaped_spectra;

    for (int photon_idx = 0; photon_idx < n_local; ++photon_idx) {
        Photon& phot = p.data[photon_idx];
        if (phot.escaped) {
            ++escaped_local;
            escaped_spectra.push_back(phot.x);
        }
        else ++carry_local;
    }

    /* remove escapees */

    for (int i = n_local - 1; i > -1; --i) {
        if (p.data[i].escaped) p.remove_photon(i);
    }

    /* global diagnostics from all ranks */

    int escaped_global = escaped_local;
    int carry_global = carry_local;
    long long scatters_global = scatters_local;
    double escaped_energy_global = escaped_energy_local;

    auto end = std::chrono::high_resolution_clock::now();
    double runtime_local = std::chrono::duration<double>(end - start_time).count();
    double runtime_max = runtime_local;

    double injected_global = injected_local;

    #ifdef PARALLEL
        MPI_Allreduce(&escaped_local, &escaped_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&carry_local, &carry_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&scatters_local, &scatters_global, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&runtime_local, &runtime_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&escaped_energy_local, &escaped_energy_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&injected_local, &injected_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    #endif

    /* reset to pre-emission value, then add the properly reduced total */
    g_lya_cum_photon_energy_injected = injected_before + injected_global;
    if (rank == 0) g_lya_cum_photon_energy_escaped += escaped_energy_global;

    #ifdef PARALLEL
        /* ensure all ranks have finished MC */
        MPI_Barrier(MPI_COMM_WORLD);
    #endif

    if (rank == 0) {
        double rate = (runtime_max > 0.0) ? static_cast<double>(n_active) / runtime_max: 0.0;
        double avg_scatters = (n_active > 0) ? static_cast<double>(scatters_global) / n_active: 0.0;
        std::cout << "        done: " << runtime_max << "s, "
             << escaped_global << " escaped, "
             << carry_global << " carry-over, "
             << static_cast<int>(rate) << " phot/s, "
             << avg_scatters << " avg scatters\n" << std::endl;
    }

    /* append escaped photon frequencies — each rank writes in turn */
    #if SPECTRUM_OUTPUT == TRUE
        #ifdef PARALLEL
            for (int r = 0; r < nprocs; ++r) {
                if (rank == r && !escaped_spectra.empty()) {
                    std::ofstream fout("output/spectrum.txt", std::ios::app);
                    for (double xv : escaped_spectra) fout << xv << "\n";
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        #else
            if (!escaped_spectra.empty()) {
                std::ofstream fout("output/spectrum.txt", std::ios::app);
                for (double xv : escaped_spectra) fout << xv << "\n";
            }
        #endif
    #endif

    #if POSITION_OUTPUT == TRUE
        #ifdef PARALLEL
            for (int r = 0; r < nprocs; ++r) {
                if (rank == r) {
                    std::ofstream fout("output/positions.txt", std::ios::app);
                    for (auto& buf : pos_buffers) {
                        for (auto& row : buf) {
                            fout << row[0] << " " << row[1] << " "
                                 << row[2] << " " << static_cast<int>(row[3]) << "\n";
                        }
                    }
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        #else
            std::ofstream fout("output/positions.txt", std::ios::app);
            for (auto& buf : pos_buffers) {
                for (auto& row : buf) {
                    fout << row[0] << " " << row[1] << " "
                            << row[2] << " " << static_cast<int>(row[3]) << "\n";
                }
            }
        #endif
    #endif

    #if MOMENTUM_OUTPUT == TRUE
        size_t nc = g.n_cells;
        #ifdef PARALLEL
            /* reduce momentum onto rank 0 */
            std::vector<double> mx(nc), my(nc), mz(nc);
            MPI_Reduce(g.mom_x.data(), mx.data(), nc, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(g.mom_y.data(), my.data(), nc, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(g.mom_z.data(), mz.data(), nc, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                std::ofstream fout("output/momentum_deposited.txt");
                fout << "x  y  z  momentum_x  momentum_y  momentum_z\n";
                for (int ix = 0; ix < g.nx; ix++)
                    for (int iy = 0; iy < g.ny; iy++)
                        for (int iz = 0; iz < g.nz; iz++) {
                            size_t idx = ix * g.ny * g.nz + iy * g.nz + iz;
                            fout << g.x_centers[ix] << " " << g.y_centers[iy] << " "
                                 << g.z_centers[iz] << " " << mx[idx] << " "
                                 << my[idx] << " " << mz[idx] << "\n";
                        }
            }
        #else
            std::ofstream fout("output/momentum_deposited.txt");
            fout << "x  y  z  momentum_x  momentum_y  momentum_z\n";
            for (int ix = 0; ix < g.nx; ix++)
                for (int iy = 0; iy < g.ny; iy++)
                    for (int iz = 0; iz < g.nz; iz++) {
                        size_t idx = ix * g.ny * g.nz + iy * g.nz + iz;
                        fout << g.x_centers[ix] << " " << g.y_centers[iy] << " "
                             << g.z_centers[iz] << " " << g.mom_x[idx] << " "
                             << g.mom_y[idx] << " " << g.mom_z[idx] << "\n";
                    }
        #endif
    #endif

    return scatters_global;
}
