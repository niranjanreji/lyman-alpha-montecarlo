#include "src/common.h"
#include "src/grid.cpp"
#include "src/photons.cpp"
#include "src/physics.cpp"
#include "src/monte_carlo.cpp"

int main () {
    const string fname = "input/grid.h5";

    Grid* grid = load_grid(fname);
    Photons* p = new Photons();
    
    monte_carlo(*p, *grid, 1e20, 100, true);
}