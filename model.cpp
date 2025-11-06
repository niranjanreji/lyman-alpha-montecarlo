/*
MODEL.CPP / NIRANJAN REJI
- CDF TABLE READING, INTERPOLATION
- GRID READING, CREATION FOR USE
*/

#include "common.h"
#include <algorithm>

using namespace std;
using namespace H5;

// ========== GLOBAL VARIABLE DEFINITIONS ==========
CDFTable g_table;
Grid3D g_grid;

// ========== CDF TABLE GLOBALS ==========

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

// Load CDF lookup table from HDF5 file. By default,
// File contains: x grid (1200 pts), T grid (25 pts), z grid (4600 pts), and 3D CDF table
void load_table(const string& path) {
    H5File f(path, H5F_ACC_RDONLY);

    // Read coordinate grids
    read_1d(f, "x", g_table.x, g_table.nx);    // Frequency offset grid
    read_1d(f, "T", g_table.T, g_table.nT);    // Temperature grid (log-spaced)
    read_1d(f, "z", g_table.z, g_table.nz);    // Shifted coordinate z = u - x

    // Read 3D CDF table [nx, nT, nz]
    DataSet cdf_ds = f.openDataSet("cdf_table");
    hsize_t dims[3];
    cdf_ds.getSpace().getSimpleExtentDims(dims);
    g_table.cdf.resize(dims[0] * dims[1] * dims[2]);
    cdf_ds.read(g_table.cdf.data(), PredType::NATIVE_DOUBLE);

    // Table is accurate for r in [0.001, 0.999]
    g_table.eps = 0.001;
    f.close();
}
    
// Fast PCHIP invert CDF (monotonic cubic) - optimized for repeated calls
inline double pchip_invert(const vector<double>& cdf, const vector<double>& z, double r) {
    // Binary search for interval
    int i = lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin() - 1;
    i = max(0, min(i, (int)cdf.size() - 2));

    double h = z[i+1] - z[i];
    double delta_cdf = cdf[i+1] - cdf[i];
    double d0 = 0.0, d1 = 0.0;

    if (i > 0 && i+2 < cdf.size()) {
        double h_prev = z[i] - z[i-1];
        double h_next = z[i+2] - z[i+1];
        double s_prev = (cdf[i] - cdf[i-1]) / h_prev;
        double s_curr = delta_cdf / h;
        double s_next = (cdf[i+2] - cdf[i+1]) / h_next;

        // Monotonic derivative at i
        if (s_prev * s_curr > 0.0) d0 = 0.5 * (s_prev + s_curr);
        // Monotonic derivative at i+1
        if (s_curr * s_next > 0.0) d1 = 0.5 * (s_curr + s_next);
    }

    // Hermite interpolation
    double t = (r - cdf[i]) / delta_cdf;
    double t2 = t * t;
    double t3 = t2 * t;

    // Optimized Hermite basis evaluation
    return z[i] + t * (h * d0 + t * (3.0*h - 2.0*h*d0 - h*d1 + t * (h*d0 + h*d1 - 2.0*h)));
}

// Bilinear + PCHIP inversion
double sample_cdf(double x_abs, double T, double r) {
    int ix = lower_bound(g_table.x.begin(), g_table.x.end(), x_abs) - g_table.x.begin() - 1;
    ix = max(0, min(ix, g_table.nx - 2));

    int iT = lower_bound(g_table.T.begin(), g_table.T.end(), T) - g_table.T.begin() - 1;
    iT = max(0, min(iT, g_table.nT - 2));

    double wx = (x_abs - g_table.x[ix]) / (g_table.x[ix+1] - g_table.x[ix]);
    double wT = (T - g_table.T[iT]) / (g_table.T[iT+1] - g_table.T[iT]);

    // Thread-local buffer to avoid repeated malloc
    thread_local vector<double> cdf_interp(g_table.nz);

    // Precompute bilinear weights
    const double w00 = (1.0 - wx) * (1.0 - wT);
    const double w10 = wx * (1.0 - wT);
    const double w01 = (1.0 - wx) * wT;
    const double w11 = wx * wT;

    for (int iz = 0; iz < g_table.nz; ++iz) {
        double c00 = g_table.at(ix, iT, iz);
        double c10 = g_table.at(ix+1, iT, iz);
        double c01 = g_table.at(ix, iT+1, iz);
        double c11 = g_table.at(ix+1, iT+1, iz);
        cdf_interp[iz] = w00*c00 + w10*c10 + w01*c01 + w11*c11;
    }

    double z_val = pchip_invert(cdf_interp, g_table.z, r);
    return z_val + x_abs;   // u = z + x
}

// ========== END CDF TABLE ============

// ========== 3D GRID GLOBALS ==========

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
    read_3d(f, "T", g_grid.T, g_grid.nx, g_grid.ny, g_grid.nz, PredType::NATIVE_INT);
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