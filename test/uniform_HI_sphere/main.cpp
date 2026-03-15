#ifdef PARALLEL
    #include <mpi.h>
#endif
#include <cstdio>
#include <chrono>
#include "common.h"
#include "rt_definitions.h"


double density_test_1(double x, double y, double z) {
    if (sqrt(x*x + y*y + z*z) < 3.086e18) return 0.565;
    return 0;
}

double density_test_2(double x, double y, double z) {
    return density_test_1(x, y, z) * 10.0;
}

double density_test_3(double x, double y, double z) {
    return density_test_1(x, y, z) * 100.0;
}

int main(int argc, char* argv[]) {
    #ifdef PARALLEL
        MPI_Init(&argc, &argv);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #else
        int rank = 0;
    #endif

    Grid* grid = init_grid(user_sources);
    Photons* p = new Photons();

    auto t0 = std::chrono::high_resolution_clock::now();
    if (rank == 0) std::cout << "Building grid (tau0 = 1e5)..." << std::endl;
    build_fields(grid, density_test_1, user_temperature, user_velocity);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (rank == 0) std::cout << "Grid build took " << std::chrono::duration<double>(t1 - t0).count() << " s. Running Monte Carlo." << std::endl;
    monte_carlo(*p, *grid, 1e20);
    if (rank == 0) std::rename("output/spectrum.txt", "output/1e5.txt");

    t0 = std::chrono::high_resolution_clock::now();
    if (rank == 0) std::cout << "Building grid (tau0 = 1e6)..." << std::endl;
    build_fields(grid, density_test_2, user_temperature, user_velocity);
    t1 = std::chrono::high_resolution_clock::now();
    if (rank == 0) std::cout << "Grid build took " << std::chrono::duration<double>(t1 - t0).count() << " s. Running Monte Carlo." << std::endl;
    monte_carlo(*p, *grid, 1e20);
    if (rank == 0) std::rename("output/spectrum.txt", "output/1e6.txt");

    t0 = std::chrono::high_resolution_clock::now();
    if (rank == 0) std::cout << "Building grid (tau0 = 1e7)..." << std::endl;
    build_fields(grid, density_test_3, user_temperature, user_velocity);
    t1 = std::chrono::high_resolution_clock::now();
    if (rank == 0) std::cout << "Grid build took " << std::chrono::duration<double>(t1 - t0).count() << " s. Running Monte Carlo." << std::endl;
    monte_carlo(*p, *grid, 1e20);
    if (rank == 0) std::rename("output/spectrum.txt", "output/1e7.txt");

    delete p;
    delete grid;

    #ifdef PARALLEL
        MPI_Finalize();
    #endif

    return 0;
}