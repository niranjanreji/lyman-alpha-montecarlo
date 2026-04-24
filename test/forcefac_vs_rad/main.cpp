#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "common.h"
#include "rt_definitions.h"

using namespace std;

namespace {

constexpr int kNumContiguousShells = 100;
constexpr int kNumGappedShells = 50;
constexpr double kMaxRadius = 3.09e18;
constexpr double kDt = 1e20;
constexpr double kSegmentWidth = kMaxRadius / (2.0 * kNumGappedShells - 1.0);
constexpr double kTwoShellInner = 1.0e18;  // inner radius of shell 1
constexpr double kTwoShellWidth = 2.0e17;

double shell_density(double x, double y, double z, double rmin, double rmax) {
    double r = sqrt(x * x + y * y + z * z);
    if (r < rmin || r >= rmax) return 0.0;
    return user_density(x, y, z);
}

double gapped_shell_density(double x, double y, double z) {
    double r = sqrt(x * x + y * y + z * z);
    if (r >= kMaxRadius) return 0.0;

    int segment = static_cast<int>(r / kSegmentWidth);
    if (segment % 2 != 0) return 0.0;  // in a gap

    return user_density(x, y, z);
}

}  // namespace

int main() {
    cout << "Initializing grid and photon container.\n";

    auto grid = unique_ptr<Grid>(init_grid(user_sources));
    Photons photons;

    /* --- Step 1: Full sphere RT --- 
    build_fields(grid.get(), user_density, user_temperature, user_velocity);
    monte_carlo(photons, *grid, kDt);
    std::rename("output/momentum_deposited.txt",
                "output/momentum_deposited_full.txt");
    photons.data.clear();
    */
    

    /* --- Step 2: Shell-by-shell RT (contiguous, each shell alone) --- 
    size_t nc = grid->n_cells;
    std::vector<double> sum_mx(nc, 0.0), sum_my(nc, 0.0), sum_mz(nc, 0.0);

    for (int i = 0; i < kNumContiguousShells; ++i) {
        double rmin = (static_cast<double>(i) / kNumContiguousShells) * kMaxRadius;
        double rmax = (static_cast<double>(i + 1) / kNumContiguousShells) * kMaxRadius;
        auto density_fn = [=](double x, double y, double z) {
            return shell_density(x, y, z, rmin, rmax);
        };
        photons.data.clear();
        build_fields(grid.get(), density_fn, user_temperature, user_velocity);
        monte_carlo(photons, *grid, kDt);
        for (size_t j = 0; j < nc; ++j) {
            sum_mx[j] += grid->mom_x[j];
            sum_my[j] += grid->mom_y[j];
            sum_mz[j] += grid->mom_z[j];
        }
    }
    photons.data.clear();

    {
        std::ofstream fout("output/momentum_deposited_shells_sum.txt");
        fout << "x  y  z  momentum_x  momentum_y  momentum_z\n";
        for (int ix = 0; ix < grid->nx; ++ix)
            for (int iy = 0; iy < grid->ny; ++iy)
                for (int iz = 0; iz < grid->nz; ++iz) {
                    size_t idx = ix * grid->ny * grid->nz + iy * grid->nz + iz;
                    fout << grid->x_centers[ix] << " " << grid->y_centers[iy] << " "
                         << grid->z_centers[iz] << " "
                         << sum_mx[idx] << " " << sum_my[idx] << " " << sum_mz[idx] << "\n";
                }
    }
    */
    
    /*
    // Run MC on half the sphere: kNumGappedShells shells with equal-width gaps,
    // all present simultaneously in the grid.
    cout << "Building gapped-shell fields (" << kNumGappedShells << " shells, "
         << "segment width = " << kSegmentWidth << " cm).\n";
    build_fields(grid.get(), gapped_shell_density, user_temperature, user_velocity);

    cout << "Running Monte Carlo on gapped shells.\n";
    monte_carlo(photons, *grid, kDt);

    if (std::rename("output/momentum_deposited.txt",
                    "output/momentum_deposited_gapped_simultaneous.txt") != 0) {
        cerr << "Failed to rename output file.\n";
        return 1;
    }
    */

    /* Run MC on two shells of width kSegmentWidth each, with shell 1 at kTwoShellInner
       and shell 2 at gap distances of 1x, 2x, 3x, 4x kSegmentWidth away. */
    for (int i = 1; i <= 10; ++i) {
        double s2_inner = kTwoShellInner + kSegmentWidth + kTwoShellWidth * i;
        double s2_outer = s2_inner + kSegmentWidth;
        auto density_fn = [=](double x, double y, double z) -> double {
            double r = sqrt(x * x + y * y + z * z);
            if (r > kTwoShellInner && r < kTwoShellInner + kSegmentWidth) return user_density(x, y, z);
            if (r > s2_inner && r < s2_outer) return user_density(x, y, z);
            return 0.0;
        };
        photons.data.clear();
        build_fields(grid.get(), density_fn, user_temperature, user_velocity);
        cout << "Running Monte Carlo: two shells, gap = " << i << " x shell width.\n";
        monte_carlo(photons, *grid, kDt);
        std::string outname = "output/momentum_deposited_twoshells_gap"
                              + std::to_string(i) + ".txt";
        if (std::rename("output/momentum_deposited.txt", outname.c_str()) != 0) {
            cerr << "Failed to rename output file.\n";
            return 1;
        }
    }

    cout << "Done.\n";
    return 0;
}
