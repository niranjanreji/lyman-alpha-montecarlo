// ---------------------------------------
// photon_engine.cpp - handles photons SoA
// ---------------------------------------

#include "common.h"
#include <chrono>

// emits a set of photons
void emit_photons(Photons& photons, Grid& grid, int num, Real dt) {
    // compute weight: (total photons emitted in dt) / num_packets
    // total_luminosity is in photons/sec
    double weight = (grid.total_luminosity * dt) / num;

    // use system time for seeding (truncated to 32 bits for mixing)
    auto now = std::chrono::high_resolution_clock::now();
    uint32_t time_seed = static_cast<uint32_t>(now.time_since_epoch().count());

    for (int phot = 0; phot < num; ++phot) {
        Photon *p = new Photon();

        // initialize per-photon RNG with unique seed combining time and photon index
        p->rng.seed(static_cast<uint64_t>(time_seed) * 2654435761u + phot);

        p->time = 0;
        p->weight = weight;

        // now pick source from sources
        int cells = grid.nx * grid.ny * grid.nz;
        Real r = urand(p->rng);
        
        // emit from grid (binary search CDF)
        if (r <= grid.lum_cdf[cells-1])
        {
            int left  = 0;
            int right = cells-1;

            while (left < right) {
                int mid = left + (right - left) / 2;
                if (grid.lum_cdf[mid] < r) left = mid + 1;
                else right = mid;
            }

            int cell_idx = left;
            int iz = cell_idx % grid.nz;
            int iy = (cell_idx / grid.nz) % grid.ny;
            int ix = cell_idx / (grid.ny * grid.nz);

            // sample random position in cell
            Real ux = urand(p->rng);
            Real uy = urand(p->rng);
            Real uz = urand(p->rng);

            p->pos_x    = grid.x_edges[ix] + ux*grid.dx;
            p->pos_y    = grid.y_edges[iy] + uy*grid.dy;
            p->pos_z    = grid.z_edges[iz] + uz*grid.dz; 
            p->cell_idx = cell_idx;
            p->local_sqrt_temp = grid.sqrt_temp[cell_idx];

            // following RASCAS, emit frequencies from a gaussian
            // with std dev = (nu_alpha / c) * sqrt(2kT / m_p)
            // in x units, std dev = 1, center = 0
            Real r1 = urand(p->rng);
            Real r2 = urand(p->rng);

            p->x = sqrt_2 * sqrt(-fast_log(r1)) * cos(two_pi * r2);
        }
        // emit from a point src
        else
        {
            int left  = cells;
            int right = grid.n_sources;

            while (left < right) {
                int mid = left + (right - left) / 2;
                if (grid.lum_cdf[mid] < r) left = mid + 1;
                else right = mid;
            }

            int ps_idx = left - cells;
            p->pos_x = grid.ps_posx[ps_idx];
            p->pos_y = grid.ps_posy[ps_idx];
            p->pos_z = grid.ps_posz[ps_idx];

            int ix = (int)((p->pos_x - grid.x_edges[0]) / grid.dx);
            int iy = (int)((p->pos_y - grid.y_edges[0]) / grid.dy);
            int iz = (int)((p->pos_z - grid.z_edges[0]) / grid.dz);
            int cell_idx = grid.ny*grid.nz*ix + grid.nz*iy + iz;

            p->cell_idx = cell_idx;
            p->local_sqrt_temp = grid.sqrt_temp[cell_idx];

            p->x = 0;
        }

        // pick direction uniformly
        Real u1 = urand(p->rng);
        Real u2 = urand(p->rng);

        Real cosine = 2.0*u1 - 1.0;
        Real phi    = two_pi*u2;

        Real sine = sqrt(1.0 - cosine*cosine);
        p->dir_x = sine * cos(phi);
        p->dir_y = sine * sin(phi);
        p->dir_z = cosine;

        photons.add_photon(*p);
        delete p;
    }
}