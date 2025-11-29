/*
MODEL.CPP / NIRANJAN REJI
- GRID READING, CREATION FOR USE
*/

#include "common.h"
#include <algorithm>

using namespace std;
using namespace H5;


Grid3D g_grid;

// ========== 3D GRID GLOBALS ==========

// Helper function to read 1D datasets
void read_1d(H5File& f, const char* name, vector<double>& v, int& n, 
                const DataType& dtype = PredType::NATIVE_DOUBLE) {
    DataSet ds = f.openDataSet(name);
    hsize_t dim;
    ds.getSpace().getSimpleExtentDims(&dim);
    n = dim;
    v.resize(dim);
    ds.read(v.data(), dtype);
}

// Helper function to read 1D scalar
template<typename T>
void read_scalar(H5File& f, const char* name, T& val, const DataType& dtype = PredType::NATIVE_INT) {
    DataSet ds = f.openDataSet(name);
    ds.read(&val, dtype);
}

// Helper function to read 3D dataset
template<typename T>
void read_3d(H5File& f, const char* name, vector<T>& v,
        int nx, int ny, int nz, const DataType& dtype) {
    DataSet ds = f.openDataSet(name);
    v.resize(nx * ny * nz);
    ds.read(v.data(), dtype);
}

// Grid loading function
void load_grid(const string& path) {
    H5File f(path, H5F_ACC_RDONLY);

    // Read grid dimensions
    read_scalar(f, "nx", g_grid.nx);
    read_scalar(f, "ny", g_grid.ny);
    read_scalar(f, "nz", g_grid.nz);

    // Read grid spacing
    read_scalar(f, "dx", g_grid.dx, PredType::NATIVE_DOUBLE);
    read_scalar(f, "dy", g_grid.dy, PredType::NATIVE_DOUBLE);
    read_scalar(f, "dz", g_grid.dz, PredType::NATIVE_DOUBLE);

    // Read domain size
    read_scalar(f, "Lx", g_grid.Lx, PredType::NATIVE_DOUBLE);
    read_scalar(f, "Ly", g_grid.Ly, PredType::NATIVE_DOUBLE);
    read_scalar(f, "Lz", g_grid.Lz, PredType::NATIVE_DOUBLE);

    // Read cell edges
    int n_temp;
    read_1d(f, "x_edges", g_grid.x_edges, n_temp);
    read_1d(f, "y_edges", g_grid.y_edges, n_temp);
    read_1d(f, "z_edges", g_grid.z_edges, n_temp);
    
    // Read cell centers
    read_1d(f, "x_centers", g_grid.x_centers, n_temp);
    read_1d(f, "y_centers", g_grid.y_centers, n_temp);
    read_1d(f, "z_centers", g_grid.z_centers, n_temp);
    
    // Read 3D physical fields
    read_3d(f, "sqrt_T", g_grid.sqrt_T, g_grid.nx, g_grid.ny, g_grid.nz, PredType::NATIVE_DOUBLE);
    read_3d(f, "HI", g_grid.HI, g_grid.nx, g_grid.ny, g_grid.nz, PredType::NATIVE_DOUBLE);
    read_3d(f, "vx", g_grid.vx, g_grid.nx, g_grid.ny, g_grid.nz, PredType::NATIVE_DOUBLE);
    read_3d(f, "vy", g_grid.vy, g_grid.nx, g_grid.ny, g_grid.nz, PredType::NATIVE_DOUBLE);
    read_3d(f, "vz", g_grid.vz, g_grid.nx, g_grid.ny, g_grid.nz, PredType::NATIVE_DOUBLE);
    
    f.close();
}

// ========== GRID HELPER FUNCTIONS ==========

// get_cell_indices: fast index lookup for uniform grid
void get_cell_indices(Photon& phot, int& ix, int& iy, int& iz) {
    ix = (int)((phot.pos_x - g_grid.x_edges[0]) / g_grid.dx);
    iy = (int)((phot.pos_y - g_grid.y_edges[0]) / g_grid.dy);
    iz = (int)((phot.pos_z - g_grid.z_edges[0]) / g_grid.dz);
}

// escaped: returns whether photon has escaped from simulation box
bool escaped(Photon& phot) {
    return (phot.pos_x < g_grid.x_edges[0] || phot.pos_x > g_grid.x_edges[g_grid.nx] ||
            phot.pos_y < g_grid.y_edges[0] || phot.pos_y > g_grid.y_edges[g_grid.ny] ||
            phot.pos_z < g_grid.z_edges[0] || phot.pos_z > g_grid.z_edges[g_grid.nz]);
}

// ========== END 3D GRID ============