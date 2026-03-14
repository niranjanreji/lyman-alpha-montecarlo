/* photons.cpp — photon packet emission and initialization.
 * creates new photon packets from grid cells and point sources,
 * sampling positions, directions, and frequencies.
 *
 * Niranjan Reji, Raman Research Institute, March 2026
 * assisted by Claude (Anthropic) */

#include <chrono>
#include "common.h"
#include "pluto_interface.h"
#include <rt_definitions.h>
#ifdef PARALLEL
    #include <mpi.h>
#endif

/**
 * @brief emit a set of photon packets into the simulation.
 * @param photons   collection to append new photons to
 * @param grid      grid containing physical fields and source info
 * @param total_num total number of packets to emit across all ranks
 * @param rank_num  number of packets this rank should emit
 * @param dt        timestep [s], used to compute packet weight
 */
void emit_photons(Photons& photons, Grid& grid, int total_num, int rank_num, double dt) {
    static int identifier = 0;

    /* compute weight: total photons in dt step / number of packets */
    double weight = (grid.total_luminosity * dt) / total_num;

    /* use system time for seeding, truncate to 32bits for mixing */
    auto now = std::chrono::high_resolution_clock::now();
    uint32_t time_seed = static_cast<uint32_t>(now.time_since_epoch().count());

    for (int phot = 0; phot < rank_num; ++phot) {
        Photon *p = new Photon();

        /* initialize per-photon RNG with unique seed using time, photon index, and rank */
        int rank = 0;
        #ifdef PARALLEL
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        #endif

        p->rng.seed(static_cast<uint64_t>(time_seed) * 2654435761u + phot * 131 + rank * 6364136223846793005u);
        p->time = 0;
        p->weight = weight;
        p->id = identifier + rank * 10000000;

        identifier++;

        /* pick emission source from luminosity CDF */
        int cells = grid.nx * grid.ny * grid.nz;
        int total_sources = cells + grid.num_point_sources;
        if (total_sources <= 0) {
            delete p;
            continue;
        }
        double r = urand(p->rng);

        /* binary search to find first entry >= r */
        auto it = std::lower_bound(grid.luminosity_CDF.begin(), grid.luminosity_CDF.end(), r);
        int j = int(it - grid.luminosity_CDF.begin());

        /* emission from grid cell */
        if (j < cells) {
            int iz = j % grid.nz;
            int iy = (j / grid.nz) % grid.ny;
            int ix = j / (grid.ny * grid.nz);

            /* sample random position within cell */
            double ux = urand(p->rng);
            double uy = urand(p->rng);
            double uz = urand(p->rng);

            p->pos_x    = grid.x_edges[ix] + ux*grid.dx;
            p->pos_y    = grid.y_edges[iy] + uy*grid.dy;
            p->pos_z    = grid.z_edges[iz] + uz*grid.dz;
            p->cell_idx = j;
            p->local_sqrt_temp = grid.sqrt_temp[j];

            /* following RASCAS, emit frequencies from a gaussian
             * with std dev = (nu_alpha / c) * sqrt(2kT / m_p)
             * in x units, std dev = 1, center = 0 */
            double r1 = urand(p->rng);
            double r2 = urand(p->rng);

            p->x = sqrt_2 * sqrt(-log(r1)) * cos(two_pi * r2);
            p->from_grid = 1;
        }
        /* emission from point source */
        else {
            int ps_idx = j - cells;

            p->pos_x = grid.ps_posx[ps_idx];
            p->pos_y = grid.ps_posy[ps_idx];
            p->pos_z = grid.ps_posz[ps_idx];

            int ix = (int)((p->pos_x - grid.x_edges[0]) / grid.dx);
            int iy = (int)((p->pos_y - grid.y_edges[0]) / grid.dy);
            int iz = (int)((p->pos_z - grid.z_edges[0]) / grid.dz);
            int cell_idx = NY * NZ * ix + NZ * iy + iz;

            p->cell_idx = cell_idx;
            p->local_sqrt_temp = grid.sqrt_temp[cell_idx];
            p->x = 0;
            p->from_grid = 0;
        }

        /* pick direction uniformly on the sphere */
        double u1 = urand(p->rng);
        double u2 = urand(p->rng);

        double cosine = 2.0*u1 - 1.0;
        double phi    = two_pi*u2;
        double sine   = sqrt(1.0 - cosine*cosine);

        p->dir_x = sine * cos(phi);
        p->dir_y = sine*sin(phi);
        p->dir_z = cosine;

        double nu_emit = double(nu_alpha) * (1.0 + double(p->x) * double(p->local_sqrt_temp) * double(vth_const) / double(c));
        if (std::isfinite(nu_emit) && nu_emit > 0.0) {
            g_lya_cum_photon_energy_injected += p->weight * double(h) * nu_emit;
        }

        photons.add_photon(*p);
        delete p;
    }
}
