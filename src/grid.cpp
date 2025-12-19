// ------------------------------
// grid.cpp - handles input file
// ------------------------------

#include "common.h"

using namespace H5;

// HDF5 type corresponding to Real
#ifdef USE_SINGLE_PRECISION
static const DataType& RealType = PredType::NATIVE_FLOAT;
#else
static const DataType& RealType = PredType::NATIVE_DOUBLE;
#endif

// helper function to read scalar attributes
template <typename T>
void read_scalar(H5File& f, const char* path, T& val,
    const DataType& dtype = PredType::NATIVE_INT) {

    string p(path);
    size_t pos = p.rfind('/');
    Group g = f.openGroup(p.substr(0, pos));
    Attribute attr = g.openAttribute(p.substr(pos + 1));
    attr.read(dtype, &val);
}

// helper function to read 1D datasets
template <typename T>
vector<T> read_1d(H5File& f, const char* name, const DataType& dtype) {

    DataSet ds = f.openDataSet(name);
    hsize_t dim;
    ds.getSpace().getSimpleExtentDims(&dim);
    vector<T> v(dim);
    ds.read(v.data(), dtype);
    return v;
}

// helper function to read 3D datasets
template <typename T>
vector<T> read_3d(H5File& f, const char* name,
    int nx, int ny, int nz, const DataType& dtype) {

    DataSet ds = f.openDataSet(name);
    vector<T> v(nx * ny * nz);
    ds.read(v.data(), dtype);
    return v;
}

Grid* load_grid(const string& path) {
    H5File f(path, H5F_ACC_RDONLY);

    Grid *grid = new Grid();

    // read grid dimensions
    read_scalar(f, "/grid/nx", grid->nx);
    read_scalar(f, "/grid/ny", grid->ny);
    read_scalar(f, "/grid/nz", grid->nz);

    // read grid spacing
    read_scalar(f, "/grid/dx", grid->dx, RealType);
    read_scalar(f, "/grid/dy", grid->dy, RealType);
    read_scalar(f, "/grid/dz", grid->dz, RealType);

    // read domain size
    read_scalar(f, "/grid/Lx", grid->Lx, RealType);
    read_scalar(f, "/grid/Ly", grid->Ly, RealType);
    read_scalar(f, "/grid/Lz", grid->Lz, RealType);

    // read cell edges
    grid->x_edges = read_1d<Real>(f, "/grid/x_edges", RealType);
    grid->y_edges = read_1d<Real>(f, "/grid/y_edges", RealType);
    grid->z_edges = read_1d<Real>(f, "/grid/z_edges", RealType);

    // read physical fields
    grid->sqrt_temp = read_3d<uint16_t>(f, "/fields/sqrt_temp", grid->nx, grid->ny, grid->nz, PredType::NATIVE_UINT16);
    grid->hi = read_3d<Real>(f, "/fields/n_HI", grid->nx, grid->ny, grid->nz, RealType);
    grid->vx = read_3d<Real>(f, "/fields/vx", grid->nx, grid->ny, grid->nz, RealType);
    grid->vy = read_3d<Real>(f, "/fields/vy", grid->nx, grid->ny, grid->nz, RealType);
    grid->vz = read_3d<Real>(f, "/fields/vz", grid->nx, grid->ny, grid->nz, RealType);

    // read luminosity data for CDF construction
    auto grid_lum = read_3d<double>(f, "/fields/nphot", grid->nx, grid->ny, grid->nz, PredType::NATIVE_DOUBLE);
    double grid_luminosity;
    read_scalar(f, "/fields/grid_luminosity", grid_luminosity, PredType::NATIVE_DOUBLE);

    // read point source data
    grid->ps_posx = read_1d<Real>(f, "/sources/ps_posx", RealType);
    grid->ps_posy = read_1d<Real>(f, "/sources/ps_posy", RealType);
    grid->ps_posz = read_1d<Real>(f, "/sources/ps_posz", RealType);

    read_scalar(f, "/sources/num", grid->n_sources, PredType::NATIVE_INT);
    auto ps_lum = read_1d<double>(f, "/sources/ps_luminosity", PredType::NATIVE_DOUBLE);
    double source_luminosity;
    read_scalar(f, "/sources/total_luminosity", source_luminosity, PredType::NATIVE_DOUBLE);
    double total_luminosity = grid_luminosity + source_luminosity;

    // store total luminosity (photons/sec) for weight calculation
    grid->total_luminosity = total_luminosity;

    size_t n_cells = grid->nx * grid->ny * grid->nz;

    // convert grid_lum, ps_lum to CDF and place in grid
    for (size_t i = 0; i < n_cells; ++i) {
        grid_lum[i] = grid_lum[i] / total_luminosity;
        if (i > 0) grid_lum[i] += grid_lum[i-1];
    }
    for (int i = 0; i < grid->n_sources; ++i) {
        ps_lum[i] = ps_lum[i] / total_luminosity;
        if (i > 0) ps_lum[i] += ps_lum[i-1];
    }

    // combine grid_lum and ps_lum into lum_cdf
    grid->lum_cdf.resize(n_cells + grid->n_sources);
    for (size_t i = 0; i < n_cells; ++i) {
        grid->lum_cdf[i] = Real(grid_lum[i]);
    }
    for (int i = 0; i < grid->n_sources; ++i) {
        grid->lum_cdf[n_cells + i] = Real(ps_lum[i] + grid_lum[n_cells - 1]);
    }

    // set up momentum grid
    grid->mom_x.assign(n_cells, Real(0.0));
    grid->mom_y.assign(n_cells, Real(0.0));
    grid->mom_z.assign(n_cells, Real(0.0));

    return grid;
}