/*
MODEL.CU / NIRANJAN REJI
- GRID READING, CREATION FOR USE
*/

#include "common.cuh"

using namespace std;
using namespace H5;


// helper to create texture object from device pointer
// -> reads are faster for read-only texture memory
template <typename T>
cudaTextureObject_t create_texture(T* d_ptr, size_t num_elements) {

    // describes what memory resource the texture will read from
    // resType - contiguous in memory, pointer to data is d_ptr,
    // its sizeInBytes long, and each element is res.linear.desc in size
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_ptr;
    resDesc.res.linear.sizeInBytes = num_elements * sizeof(T);
    resDesc.res.linear.desc = cudaCreateChannelDesc<T>>();

    // the texture mem is read only, so this sets that explicitly
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texObj;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    return texObj;
}


// helper function to read 1d data into pinned memory
void read_1d_pinned(H5File& f, const char* name, double*& pinned_ptr, int& n) {
    DataSet ds = f.openDataSet(name);
    hsize_t dim;
    ds.getSpace().getSimpleExtentDims(&dim);
    n = dim;
    
    // Allocate pinned memory for fast GPU transfer
    cudaHostAlloc(&pinned_ptr, dim * sizeof(double), cudaHostAllocDefault);
    ds.read(pinned_ptr, PredType::NATIVE_DOUBLE);
}


// helper function to read scalars
template <typename T>
void read_scalar(H5File& f, const char* name, T& val, const DataType& dtype = PredType::NATIVE_INT) {
    DataSet ds = f.openDataSet(name);
    ds.read(&val, dtype);
}

// helper function to read 3d dataset into pinned memory
template<typename T>
void read_3d_pinned(H5File& f, const char* name, T*& pinned_ptr, 
    int nx, int ny, int nz, const DataType& dtype) {
    
    DataSet ds = f.openDataSet(name);
    size_t num_elements = nx * ny * nz;

    // allocate pinned mem
    cudaHostAlloc(&pinned_ptr, num_elements * sizeof(T), cudaHostAllocDefault);
    ds.read(pinned_ptr, dtype);
}

// main grid loading function
__host__ Grid3D load_grid(const string& path) {
    Grid3D grid;
    H5File f(path, H5F_ACC_RDONLY);

    // read dimensions of grid to host (temporary variables)
    int nx, ny, nz;
    double dx, dy, dz, Lx, Ly, Lz;

    read_scalar(f, "nx", nx);
    read_scalar(f, "ny", ny);
    read_scalar(f, "nz", nz);

    // read grid spacing
    read_scalar(f, "dx", dx, PredType::NATIVE_DOUBLE);
    read_scalar(f, "dy", dy, PredType::NATIVE_DOUBLE);
    read_scalar(f, "dz", dz, PredType::NATIVE_DOUBLE);

    // read domain size
    read_scalar(f, "Lx", Lx, PredType::NATIVE_DOUBLE);
    read_scalar(f, "Ly", Ly, PredType::NATIVE_DOUBLE);
    read_scalar(f, "Lz", Lz, PredType::NATIVE_DOUBLE);

    // Copy grid metadata to CONSTANT memory for ultra-fast access
    cudaMemcpyToSymbol(g_nx, &nx, sizeof(int));
    cudaMemcpyToSymbol(g_ny, &ny, sizeof(int));
    cudaMemcpyToSymbol(g_nz, &nz, sizeof(int));
    cudaMemcpyToSymbol(g_dx, &dx, sizeof(double));
    cudaMemcpyToSymbol(g_dy, &dy, sizeof(double));
    cudaMemcpyToSymbol(g_dz, &dz, sizeof(double));

    // create cuda stream for async ops
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Read edge and center arrays
    double *h_x_edges, *h_y_edges, *h_z_edges;
    double *h_x_centers, *h_y_centers, *h_z_centers;
    int n_temp;

    read_1d_pinned(f, "x_edges", h_x_edges, n_temp);
    read_1d_pinned(f, "y_edges", h_y_edges, n_temp);
    read_1d_pinned(f, "z_edges", h_z_edges, n_temp);
    read_1d_pinned(f, "x_centers", h_x_centers, n_temp);
    read_1d_pinned(f, "y_centers", h_y_centers, n_temp);
    read_1d_pinned(f, "z_centers", h_z_centers, n_temp);

    // Extract domain bounds and copy to constant memory (for boundary checks)
    double x_min = h_x_edges[0], x_max = h_x_edges[nx];
    double y_min = h_y_edges[0], y_max = h_y_edges[ny];
    double z_min = h_z_edges[0], z_max = h_z_edges[nz];

    cudaMemcpyToSymbol(g_x_min, &x_min, sizeof(double));
    cudaMemcpyToSymbol(g_y_min, &y_min, sizeof(double));
    cudaMemcpyToSymbol(g_z_min, &z_min, sizeof(double));
    cudaMemcpyToSymbol(g_x_max, &x_max, sizeof(double));
    cudaMemcpyToSymbol(g_y_max, &y_max, sizeof(double));
    cudaMemcpyToSymbol(g_z_max, &z_max, sizeof(double));

    // Allocate device memory for edge/center arrays
    double *d_x_edges, *d_y_edges, *d_z_edges;
    double *d_x_centers, *d_y_centers, *d_z_centers;

    cudaMalloc(&d_x_edges, (nx + 1) * sizeof(double));
    cudaMalloc(&d_y_edges, (ny + 1) * sizeof(double));
    cudaMalloc(&d_z_edges, (nz + 1) * sizeof(double));
    cudaMalloc(&d_x_centers, nx * sizeof(double));
    cudaMalloc(&d_y_centers, ny * sizeof(double));
    cudaMalloc(&d_z_centers, nz * sizeof(double));

    // Async copy edge/center arrays to device
    cudaMemcpyAsync(d_x_edges, h_x_edges, (nx + 1) * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_y_edges, h_y_edges, (ny + 1) * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_z_edges, h_z_edges, (nz + 1) * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_x_centers, h_x_centers, nx * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_y_centers, h_y_centers, ny * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_z_centers, h_z_centers, nz * sizeof(double), cudaMemcpyHostToDevice, stream);

    // read 3D fields (will be bound to texture memory)
    size_t num_cells = nx * ny * nz;

    // read arrays into host mem first
    int *h_sqrt_T;
    double *h_HI, *h_vx, *h_vy, *h_vz;

    read_3d_pinned(f, "sqrt_T", h_sqrt_T, nx, ny, nz, PredType::NATIVE_INT);
    read_3d_pinned(f, "HI", h_HI, nx, ny, nz, PredType::NATIVE_DOUBLE);
    read_3d_pinned(f, "vx", h_vx, nx, ny, nz, PredType::NATIVE_DOUBLE);
    read_3d_pinned(f, "vy", h_vy, nx, ny, nz, PredType::NATIVE_DOUBLE);
    read_3d_pinned(f, "vz", h_vz, nx, ny, nz, PredType::NATIVE_DOUBLE);

    // allocate device memory for these fields
    int* d_sqrt_T;
    double *d_HI, *d_vx, *d_vy, *d_vz;
    
    cudaMalloc(&d_sqrt_T, num_cells * sizeof(int));
    cudaMalloc(&d_HI, num_cells * sizeof(double));
    cudaMalloc(&d_vx, num_cells * sizeof(double));
    cudaMalloc(&d_vy, num_cells * sizeof(double));
    cudaMalloc(&d_vz, num_cells * sizeof(double));

    // async copy to device
    cudaMemcpyAsync(d_sqrt_T, h_sqrt_T, num_cells * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_HI, h_HI, num_cells * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_vx, h_vx, num_cells * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_vy, h_vy, num_cells * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_vz, h_vz, num_cells * sizeof(double), cudaMemcpyHostToDevice, stream);

    // wait for streams to complete
    cudaStreamSynchronize(stream);

    // create texture objects for 3D fields (for faster reads in mc)
    grid.sqrt_T = create_texture<int>(d_sqrt_T, num_cells);
    grid.HI = create_texture<double>(d_HI, num_cells);
    grid.vx = create_texture<double>(d_vx, num_cells);
    grid.vy = create_texture<double>(d_vy, num_cells);
    grid.vz = create_texture<double>(d_vz, num_cells);

    // create texture objects for edge/center arrays
    grid.tex_x_edges = create_texture<double>(d_x_edges, nx + 1);
    grid.tex_y_edges = create_texture<double>(d_y_edges, ny + 1);
    grid.tex_z_edges = create_texture<double>(d_z_edges, nz + 1);
    grid.tex_x_centers = create_texture<double>(d_x_centers, nx);
    grid.tex_y_centers = create_texture<double>(d_y_centers, ny);
    grid.tex_z_centers = create_texture<double>(d_z_centers, nz);

    // free the host memory we used earlier
    cudaFreeHost(h_x_edges); cudaFreeHost(h_y_edges); cudaFreeHost(h_z_edges);
    cudaFreeHost(h_x_centers); cudaFreeHost(h_y_centers); cudaFreeHost(h_z_centers);
    cudaFreeHost(h_sqrt_T); cudaFreeHost(h_HI);
    cudaFreeHost(h_vx); cudaFreeHost(h_vy); cudaFreeHost(h_vz);

    cudaStreamDestroy(stream);
    f.close();

    return grid;
}


// de-alloc function for the texture + other device memory
__host__ void free_grid(Grid3D& grid) {
    // Destroy 3D field texture objects
    cudaDestroyTextureObject(grid.sqrt_T);
    cudaDestroyTextureObject(grid.HI);
    cudaDestroyTextureObject(grid.vx);
    cudaDestroyTextureObject(grid.vy);
    cudaDestroyTextureObject(grid.vz);

    // Destroy edge/center texture objects
    cudaDestroyTextureObject(grid.tex_x_edges);
    cudaDestroyTextureObject(grid.tex_y_edges);
    cudaDestroyTextureObject(grid.tex_z_edges);
    cudaDestroyTextureObject(grid.tex_x_centers);
    cudaDestroyTextureObject(grid.tex_y_centers);
    cudaDestroyTextureObject(grid.tex_z_centers);
}


// locates photon on grid - uses constant memory for ultra-fast access
__device__ void get_cell_indices(const Photon& phot, const Grid3D& grid, int& ix, int& iy, int& iz) {
    ix = (int)((phot.pos_x - g_x_min) / g_dx);
    iy = (int)((phot.pos_y - g_y_min) / g_dy);
    iz = (int)((phot.pos_z - g_z_min) / g_dz);
}


// checks if photon has escaped from grid - uses constant memory
__device__ bool escaped(const Photon& phot, const Grid3D& grid) {
    return (phot.pos_x < g_x_min || phot.pos_x > g_x_max ||
            phot.pos_y < g_y_min || phot.pos_y > g_y_max ||
            phot.pos_z < g_z_min || phot.pos_z > g_z_max);
}