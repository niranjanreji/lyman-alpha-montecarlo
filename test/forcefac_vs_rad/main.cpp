#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "common.h"
#include "rt_definitions.h"

using namespace std;

namespace {

constexpr int kNumShells = 100;
constexpr double kMaxRadius = 3.1e18;
constexpr double kDt = 1e20;

double g_shell_rmin = 0.0;
double g_shell_rmax = 0.0;

double shell_density(double x, double y, double z) {
    double r = sqrt(x*x + y*y + z*z);
    if (r < g_shell_rmin || r >= g_shell_rmax) return 0.0;
    return user_density(x, y, z);
}

void set_shell_bounds(double rmin, double rmax) {
    g_shell_rmin = rmin;
    g_shell_rmax = rmax;
}

bool rename_output_file(const string& from, const string& to) {
    if (std::rename(from.c_str(), to.c_str()) == 0) return true;

    cerr << "Failed to rename " << from << " to " << to << '\n';
    return false;
}

bool remove_output_file(const string& path) {
    if (std::remove(path.c_str()) == 0) return true;
    if (errno == ENOENT) return true;

    cerr << "Failed to remove " << path << '\n';
    return false;
}

bool clear_previous_outputs() {
    namespace fs = std::filesystem;
    const fs::path output_dir("output");

    if (!fs::exists(output_dir)) return true;

    for (const fs::directory_entry& entry : fs::directory_iterator(output_dir)) {
        if (!entry.is_regular_file()) continue;

        const string name = entry.path().filename().string();
        if (name == "momentum_deposited.txt" ||
            name == "momentum_deposited_full.txt" ||
            name == "momentum_deposited_shell_sum.txt" ||
            name.rfind("momentum_deposited_shell_", 0) == 0) {
            if (std::remove(entry.path().c_str()) != 0) {
                cerr << "Failed to remove " << entry.path() << '\n';
                return false;
            }
        }
    }

    return true;
}

void accumulate_momentum(const Grid& grid, vector<double>& sum_x,
                         vector<double>& sum_y, vector<double>& sum_z) {
    for (size_t idx = 0; idx < grid.n_cells; ++idx) {
        sum_x[idx] += grid.mom_x[idx];
        sum_y[idx] += grid.mom_y[idx];
        sum_z[idx] += grid.mom_z[idx];
    }
}

bool write_momentum_output(const Grid& grid, const vector<double>& mom_x,
                           const vector<double>& mom_y, const vector<double>& mom_z,
                           const string& path) {
    ofstream fout(path);
    if (!fout) {
        cerr << "Failed to open " << path << " for writing\n";
        return false;
    }

    fout << "x  y  z  momentum_x  momentum_y  momentum_z\n";
    for (int ix = 0; ix < grid.nx; ++ix) {
        for (int iy = 0; iy < grid.ny; ++iy) {
            for (int iz = 0; iz < grid.nz; ++iz) {
                size_t idx = ix * grid.ny * grid.nz + iy * grid.nz + iz;
                fout << grid.x_centers[ix] << " " << grid.y_centers[iy] << " "
                     << grid.z_centers[iz] << " " << mom_x[idx] << " "
                     << mom_y[idx] << " " << mom_z[idx] << "\n";
            }
        }
    }

    return true;
}

}  // namespace

int main() {
    cout << "Initializing grid and photon container.\n";
    if (!clear_previous_outputs()) return 1;

    auto grid = unique_ptr<Grid>(init_grid(user_sources));
    Photons photons;
    vector<double> shell_sum_x(grid->n_cells, 0.0);
    vector<double> shell_sum_y(grid->n_cells, 0.0);
    vector<double> shell_sum_z(grid->n_cells, 0.0);

    /* first, full RT */
    cout << "Building full-grid fields.\n";
    build_fields(grid.get(), user_density, user_temperature, user_velocity);
    cout << "Running full-grid Monte Carlo.\n";
    monte_carlo(photons, *grid, kDt);

    cout << "Saving full-grid momentum output.\n";
    if (!rename_output_file("output/momentum_deposited.txt",
                            "output/momentum_deposited_full.txt")) {
        return 1;
    }

    /* now, shell by shell RT - split grid into 100 cells,
     * each with the same optical depth radially */
    for (int i = 1; i <= kNumShells; ++i) {
        double rmax = (static_cast<double>(i) / kNumShells) * kMaxRadius;
        double rmin = (static_cast<double>(i - 1) / kNumShells) * kMaxRadius;

        cout << "Preparing shell " << i << "/" << kNumShells
             << " with r in [" << rmin << ", " << rmax << ").\n";
        set_shell_bounds(rmin, rmax);

        /* start each shell run with a fresh photon population */
        cout << "Clearing carry-over photons.\n";
        photons.data.clear();

        cout << "Building shell fields.\n";
        build_fields(grid.get(), shell_density, user_temperature, user_velocity);
        cout << "Running shell Monte Carlo.\n";
        monte_carlo(photons, *grid, kDt);

        cout << "Updating cumulative shell momentum output.\n";
        accumulate_momentum(*grid, shell_sum_x, shell_sum_y, shell_sum_z);
        if (!write_momentum_output(*grid, shell_sum_x, shell_sum_y, shell_sum_z,
                                   "output/momentum_deposited_shell_sum.txt")) {
            return 1;
        }

        cout << "Removing temporary shell momentum output.\n";
        if (!remove_output_file("output/momentum_deposited.txt")) {
            return 1;
        }
    }

    cout << "Done.\n";
    return 0;
}
