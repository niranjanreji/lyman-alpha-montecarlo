/*
TEST_SPHERE.CPP / NIRANJAN REJI
*/

#include <string>
#include <cstdlib>

#include "../../src/common.h"
#include "../../src/grid.cpp"
#include "../../src/physics.cpp"
#include "../../src/photons.cpp"
#include "../../src/monte_carlo.cpp"

using namespace std;

void create_grid(double num_density) {
    string cmd = "cd ../../input && python3 grid_create.py grid.h5 --density " + to_string(num_density);
    int ret = system(cmd.c_str());
    if (ret != 0) throw runtime_error("grid_create.py failed");
}

int main() {
    // r = 3e18 cm, n = 0.565 -> tau = 1e5
    create_grid(0.565);
    Grid* grid = load_grid("../../input/grid.h5");
    Photons* p = new Photons();

    monte_carlo(*p, *grid, 1e30, 50000, true);
    string cmd = "mv output/spectrum.txt 1e5.txt";
    int ret = system(cmd.c_str());
    delete p;
    delete grid;

    create_grid(5.65);
    grid = load_grid("../../input/grid.h5");
    p = new Photons();

    monte_carlo(*p, *grid, 1e30, 50000, true);
    cmd = "mv output/spectrum.txt 1e6.txt";
    ret = system(cmd.c_str());
    delete p;
    delete grid;

    create_grid(56.5);
    grid = load_grid("../../input/grid.h5");
    p = new Photons();

    monte_carlo(*p, *grid, 1e30, 50000, true);
    cmd = "mv output/spectrum.txt 1e7.txt";
    ret = system(cmd.c_str());
    delete p;
    delete grid;

    return 0;
}