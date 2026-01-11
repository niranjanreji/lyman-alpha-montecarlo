#include "src/common.h"
#include "src/grid.cpp"
#include "src/photons.cpp"
#include "src/physics.cpp"
#include "src/monte_carlo.cpp"

int main () {
    const string fname = "input/grid.h5";

    Grid* grid = load_grid(fname);
    Photons* p = new Photons();

    double total_time = 0;
    while (total_time < 1e12) {
        monte_carlo(*p, *grid, 1e9, 10, true);
        total_time += 1e9;
        cout << "Current time: " << total_time << " s" << endl;
    }
}