#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../timer.h"

#define SOFTENING 0.01f 
#define USE_CUTOFF 1
#define RCUT      100.0f  
#define RCUT_SQR  (RCUT * RCUT)
// Tuned GPU parameters
#define THREADS_PER_BLOCK 256
#define TILE_SIZE         256      // match block size
#define NUM_STREAMS       16
#define MAX_BLOCKS_PER_SYS 32

// Inaccurate version parameters
#define NEAR_RADIUS 128   // how many neighbor indices on each side are "exact"

#define ABS(val)    ((val)<0.0 ? (-(val)) : (val))

#define IS_NULL(ptr) (((ptr) == (NULL)) ? 1 : 0)
#define IS_CUDA_MALLOC_OK(err) (((err) == (cudaSuccess)) ? (1) : (0))
#define FREE_MEM(ptr) (((ptr) != (NULL)) ? (free(ptr)) : ((void)(ptr))) // Free CPU mem

// SoA device arrays
float *d_x  = NULL, *d_y  = NULL, *d_z  = NULL;
float *d_vx = NULL, *d_vy = NULL, *d_vz = NULL;

// Texture objects for positions (read-only, exact)
cudaTextureObject_t tex_x = 0;
cudaTextureObject_t tex_y = 0;
cudaTextureObject_t tex_z = 0;

// Coarse (approximate) position field in 1D, for far interactions
float *d_x_coarse = NULL, *d_y_coarse = NULL, *d_z_coarse = NULL;
int    coarse_stride = 2048;    // sample every N bodies (tune this)
int    coarse_size   = 0;    // number of coarse samples globally

cudaTextureObject_t tex_x_coarse = 0;
cudaTextureObject_t tex_y_coarse = 0;
cudaTextureObject_t tex_z_coarse = 0;



// Check for run time error and free allocated memory if you detect failure
#define CHECK_CUDA_FAIL(err) \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s (%d) in %s, line %i\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
        cudaFreeHost(h_data.all_coords); \
        cudaFreeHost(h_data.all_velocities); \
        FREE_MEM(omp_out.all_coords); \
        FREE_MEM(omp_out.all_velocities); \
        cudaFreeHost(h_device_out.all_coords); \
        cudaFree(d_data.all_coords); \
        cudaFree(d_data.all_velocities); \
        cudaDestroyTextureObject(tex_x); \
        cudaDestroyTextureObject(tex_y); \
        cudaDestroyTextureObject(tex_z); \
        cudaDestroyTextureObject(tex_x_coarse); \
        cudaDestroyTextureObject(tex_y_coarse); \
        cudaDestroyTextureObject(tex_z_coarse); \
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_z); \
        cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz); \
        cudaFree(d_x_coarse); cudaFree(d_y_coarse); cudaFree(d_z_coarse); \
        cudaDeviceReset(); \
        exit(1); \
    }

// compile time flag
#ifdef DEBUG
    int debug = 1;
#else
    int debug = 0;
#endif

typedef struct {
    float x, y, z;
} coords_T;

typedef struct {
    float vx, vy, vz;
} velocity_T;

typedef struct {
    coords_T *all_coords;
    velocity_T *all_velocities;
} Universe;

Universe h_data, h_device_out, omp_out, d_data;
cudaStream_t streams[NUM_STREAMS];

// Global flag to switch between accurate and inaccurate kernels
int use_inaccurate = 1;

/* ==================== CPU CODE (UNCHANGED) ==================== */

/* Update a single galaxy. Parameters:
    - array of bodies
    - time step
    - number of bodies
*/
void bodyForce(coords_T * coor_p, velocity_T * vel_p, float dt, int n) {
    int i, j;
    float Fx, Fy, Fz, dx, dy, dz, distSqr, invDist, invDist3;

    for (i = 0; i < n; i++) {
        Fx = 0.0f;
        Fy = 0.0f;
        Fz = 0.0f;
        
        coords_T curr = coor_p[i];
        for (j = 0; j < n; j++) {
            dx = coor_p[j].x - coor_p[i].x;
            dy = coor_p[j].y - coor_p[i].y;
            dz = coor_p[j].z - coor_p[i].z;
            distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            invDist = 1.0f / sqrtf(distSqr);
            invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        vel_p[i].vx += dt * Fx;
        vel_p[i].vy += dt * Fy;
        vel_p[i].vz += dt * Fz;
    }
}

/* Integrate positions.
    - array of bodies
    - time step
    - number of bodies
*/
void integrate(coords_T * coor_p, velocity_T * vel_p, float dt, int n) {
    int i;
    
    for (i = 0; i < n; i++) {
        coor_p[i].x += vel_p[i].vx * dt;
        coor_p[i].y += vel_p[i].vy * dt;
        coor_p[i].z += vel_p[i].vz * dt;
    }
}

/* OMP CODE */
void run_omp(int num_systems, int nIters, int bodies_per_system, float dt) {
    double totalTime;
    int iter, sys;
    double interactions_per_system, total_interactions;

    printf("\n----------------------------\n");
    printf("Running parallel CPU simulation for %d systems...\n",
           num_systems);

    totalTime = 0.0;
    // CPU code
    StartTimer();

    /* Time-steps */        
    for (iter = 1; iter <= nIters; iter++) {
        /* Galaxies */    
        #pragma omp parallel for
        for (sys = 0; sys < num_systems; sys++) {
            /* Calculate offset for the galaxy */
            // private pointer for each thread. One thread per system
            coords_T *coords_ptr = &(omp_out.all_coords[sys * bodies_per_system]);
            velocity_T *veloc_ptr = &(omp_out.all_velocities[sys * bodies_per_system]);
            
            /* Compute forces & integrate for the galaxy */
            bodyForce(coords_ptr, veloc_ptr, dt, bodies_per_system);
            integrate(coords_ptr, veloc_ptr, dt, bodies_per_system);
        }
    }
    totalTime = GetTimer() / 1000.0;

    /* Metrics calculation */
    interactions_per_system = (double) bodies_per_system * bodies_per_system;
    total_interactions = interactions_per_system * num_systems * nIters;
    printf("Total Time CPU: %.3f seconds\n", totalTime);
    printf("Average Throughput CPU: %0.3f Billion Interactions / second\n\n",
           1e-9 * total_interactions / totalTime);

    /* Dump final state of System 0, Body 0 and 1 for verification comparison */
    printf("Final position of System 0, Body 0: %.4f, %.4f, %.4f\n",
           omp_out.all_coords[0].x, omp_out.all_coords[0].y, omp_out.all_coords[0].z);
    printf("Final position of System 0, Body 1: %.4f, %.4f, %.4f\n",
           omp_out.all_coords[1].x, omp_out.all_coords[1].y, omp_out.all_coords[1].z);
    printf("----------------------------\n");
}

/* ==================== GPU KERNELS (SoA + TEXTURE) ==================== */

// Pack AoS coords_T into SoA (x,y,z) for one system (offset into global arrays)
__global__ void pack_positions_kernel(const coords_T *aos,
                                      float *x, float *y, float *z,
                                      int n, int offset)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;
    for (int i = tid; i < n; i += stride) {
        coords_T c = aos[offset + i];
        x[offset + i] = c.x;
        y[offset + i] = c.y;
        z[offset + i] = c.z;
    }
}

// Pack AoS velocity_T into SoA (vx,vy,vz)
__global__ void pack_velocities_kernel(const velocity_T *aos,
                                       float *vx, float *vy, float *vz,
                                       int n, int offset)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;
    for (int i = tid; i < n; i += stride) {
        velocity_T v = aos[offset + i];
        vx[offset + i] = v.vx;
        vy[offset + i] = v.vy;
        vz[offset + i] = v.vz;
    }
}

// Unpack SoA (x,y,z) back into AoS coords_T
__global__ void unpack_positions_kernel(coords_T *aos,
                                        const float *x, const float *y, const float *z,
                                        int n, int offset)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;
    for (int i = tid; i < n; i += stride) {
        coords_T c;
        c.x = x[offset + i];
        c.y = y[offset + i];
        c.z = z[offset + i];
        aos[offset + i] = c;
    }
}

// Build coarse field: sample every 'stride' bodies
__global__ void build_coarse_field(const float *x, const float *y, const float *z,
                                   float *x_coarse, float *y_coarse, float *z_coarse,
                                   int total_bodies, int stride)
{
    int cid = blockIdx.x * blockDim.x + threadIdx.x;  // coarse index
    int cstride = gridDim.x * blockDim.x;
    for (int i = cid; i * stride < total_bodies; i += cstride) {
        int gi = i * stride;
        x_coarse[i] = x[gi];
        y_coarse[i] = y[gi];
        z_coarse[i] = z[gi];
    }
}

// ACCURATE: SoA force kernel with texture reads (point sampling, exact)
__global__ void update_velocities(cudaTextureObject_t tex_x,
                                  cudaTextureObject_t tex_y,
                                  cudaTextureObject_t tex_z,
                                  float *__restrict__ vx,
                                  float *__restrict__ vy,
                                  float *__restrict__ vz,
                                  int bodies_per_system,
                                  int system_offset,    // offset in global arrays
                                  float dt) 
{
    __shared__ float sx[TILE_SIZE];
    __shared__ float sy[TILE_SIZE];
    __shared__ float sz[TILE_SIZE];

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;

    const int tiles_per_sys = (bodies_per_system + TILE_SIZE - 1) / TILE_SIZE;

    for (int i = tid; i < bodies_per_system; i += stride) {
        int gi = system_offset + i;  // global index in [0, total_bodies)
        float xi = tex1Dfetch<float>(tex_x, gi);
        float yi = tex1Dfetch<float>(tex_y, gi);
        float zi = tex1Dfetch<float>(tex_z, gi);

        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

        for (int tile = 0; tile < tiles_per_sys; tile++) {
            int tile_offset = tile * TILE_SIZE;

            // global index of first body in this tile
            int gbase = system_offset + tile_offset;

            int load_idx = tile_offset + threadIdx.x;  // local index 0..bodies_per_system-1
            if (load_idx < bodies_per_system) {
                int gload = gbase + threadIdx.x;        // global index for that body
                sx[threadIdx.x] = tex1Dfetch<float>(tex_x, gload);
                sy[threadIdx.x] = tex1Dfetch<float>(tex_y, gload);
                sz[threadIdx.x] = tex1Dfetch<float>(tex_z, gload);
            }
            __syncthreads();

            int tile_end = min(TILE_SIZE, bodies_per_system - tile_offset);

            #pragma unroll 8
            for (int j = 0; j < tile_end; j++) {
                float dx = sx[j] - xi;
                float dy = sy[j] - yi;
                float dz = sz[j] - zi;
            
            #if USE_CUTOFF
                // raw distance^2 without softening
                float rawDistSqr = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
                if (rawDistSqr > RCUT_SQR) {
                    // skip this interaction (too far)
                    continue;
                }
                float distSqr = rawDistSqr + SOFTENING;
            #else
                float distSqr = fmaf(dx, dx, fmaf(dy, dy, dz * dz)) + SOFTENING;
            #endif
            
                float invDist = rsqrtf(distSqr);
                float f       = invDist * invDist * invDist;
            
                Fx += dx * f;
                Fy += dy * f;
                Fz += dz * f;
            }
            __syncthreads();
        }

        vx[gi] += dt * Fx;
        vy[gi] += dt * Fy;
        vz[gi] += dt * Fz;
    }
}

// INACCURATE: Approximate force kernel using linear-filtered coarse textures
__global__ void update_velocities_inaccurate(cudaTextureObject_t tex_x,
                                             cudaTextureObject_t tex_y,
                                             cudaTextureObject_t tex_z,
                                             cudaTextureObject_t tex_xc,
                                             cudaTextureObject_t tex_yc,
                                             cudaTextureObject_t tex_zc,
                                             float *__restrict__ vx,
                                             float *__restrict__ vy,
                                             float *__restrict__ vz,
                                             int bodies_per_system,
                                             int system_offset,
                                             int coarse_stride,
                                             int coarse_size,
                                             float dt)
{
    __shared__ float sx[TILE_SIZE];
    __shared__ float sy[TILE_SIZE];
    __shared__ float sz[TILE_SIZE];

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;

    const int tiles_per_sys = (bodies_per_system + TILE_SIZE - 1) / TILE_SIZE;

    for (int i = tid; i < bodies_per_system; i += stride) {
        int gi = system_offset + i;  // global index
        float xi = tex1Dfetch<float>(tex_x, gi);
        float yi = tex1Dfetch<float>(tex_y, gi);
        float zi = tex1Dfetch<float>(tex_z, gi);

        float Fx_near = 0.0f, Fy_near = 0.0f, Fz_near = 0.0f;
        float Fx_far  = 0.0f, Fy_far  = 0.0f, Fz_far  = 0.0f;

        // --- 1) Near-field: exact, all j with |j-i| <= NEAR_RADIUS ---
        for (int tile = 0; tile < tiles_per_sys; tile++) {
            int tile_offset = tile * TILE_SIZE;
            int gbase = system_offset + tile_offset;

            int load_idx = tile_offset + threadIdx.x;
            if (load_idx < bodies_per_system) {
                int gload = gbase + threadIdx.x;
                sx[threadIdx.x] = tex1Dfetch<float>(tex_x, gload);
                sy[threadIdx.x] = tex1Dfetch<float>(tex_y, gload);
                sz[threadIdx.x] = tex1Dfetch<float>(tex_z, gload);
            }
            __syncthreads();

            int tile_end = min(TILE_SIZE, bodies_per_system - tile_offset);

            #pragma unroll 8
            for (int j = 0; j < tile_end; j++) {
                int j_local = tile_offset + j;
                if (abs(j_local - i) > NEAR_RADIUS) continue; // skip far here

                float dx = sx[j] - xi;
                float dy = sy[j] - yi;
                float dz = sz[j] - zi;

                float distSqr = fmaf(dx, dx, fmaf(dy, dy, dz * dz)) + SOFTENING;
                float invDist = rsqrtf(distSqr);
                float f       = invDist * invDist * invDist;

                Fx_near += dx * f;
                Fy_near += dy * f;
                Fz_near += dz * f;
            }
            __syncthreads();
        }

        // --- 2) Far-field: approximate via coarse field (linear filter) ---
        // Map i within [0, bodies_per_system) to coarse index range
        float scale = (float)coarse_size / (float)bodies_per_system;
        float ci_f  = (i * scale);          // fractional coarse index
        // sample a *window* of coarse samples (very rough approximation)
        int window = 8;                     // how many coarse cells we consider
        int ci0    = max(0, (int)ci_f - window/2);
        int ci1    = min(coarse_size-1, ci0 + window);

        for (int kc = ci0; kc <= ci1; kc++) {
            // sample interpolated coarse position
            // tex1D() uses float coord; with normalizedCoords=0 this is element index
            float xc = tex1D<float>(tex_xc, (float)kc);
            float yc = tex1D<float>(tex_yc, (float)kc);
            float zc = tex1D<float>(tex_zc, (float)kc);

            float dx = xc - xi;
            float dy = yc - yi;
            float dz = zc - zi;

            float distSqr = fmaf(dx, dx, fmaf(dy, dy, dz * dz)) + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float f       = invDist * invDist * invDist;

            // We treat each coarse sample as representing 'coarse_stride' bodies.
            Fx_far += dx * f * coarse_stride;
            Fy_far += dy * f * coarse_stride;
            Fz_far += dz * f * coarse_stride;
        }

        float Fx = Fx_near + Fx_far;
        float Fy = Fy_near + Fy_far;
        float Fz = Fz_near + Fz_far;

        vx[gi] += dt * Fx;
        vy[gi] += dt * Fy;
        vz[gi] += dt * Fz;
    }
}

__global__ void update_positions(float *__restrict__ x,
                                 float *__restrict__ y,
                                 float *__restrict__ z,
                                 const float *__restrict__ vx,
                                 const float *__restrict__ vy,
                                 const float *__restrict__ vz,
                                 int bodies_per_system,
                                 int system_offset,
                                 float dt) 
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;

    for (int i = tid; i < bodies_per_system; i += stride) {
        int gi = system_offset + i;
        x[gi] += vx[gi] * dt;
        y[gi] += vy[gi] * dt;
        z[gi] += vz[gi] * dt;
    }
}

/* ==================== GPU DRIVER (run_device) ==================== */

void run_device(coords_T *d_all_coords, velocity_T *d_all_velocities,
                int num_systems, int bytes_coords, int bytes_velocs,
                int nIters, int bodies_per_system, float dt) {

    double totalTime;
    cudaError_t runtime_error;
    double interactions_per_system, total_interactions;

    printf("\n----------------------------\n");
    if (use_inaccurate) {
        printf("Running Device simulation (INACCURATE: SoA + coarse texture) for %d systems...\n",
               num_systems);
    } else {
        printf("Running Device simulation (ACCURATE: SoA + texture) for %d systems...\n",
               num_systems);
    }

    int bytes_coor_system = sizeof(coords_T)   * bodies_per_system;
    int bytes_vel_system  = sizeof(velocity_T) * bodies_per_system;

    // Compute blocks from bodies_per_system, cap at MAX_BLOCKS_PER_SYS
    int blocks = (bodies_per_system + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks > MAX_BLOCKS_PER_SYS) blocks = MAX_BLOCKS_PER_SYS;

    dim3 grid_dims  (blocks, 1);
    dim3 block_dims (THREADS_PER_BLOCK, 1);

    printf("Using grid_dims.x = %d, block_dims.x = %d\n",
           grid_dims.x, block_dims.x);

    StartTimer();
    
    for (int sys = 0; sys < num_systems; sys++) {
        int stream_id = sys % NUM_STREAMS;       // Round-robin over streams
        int offset    = sys * bodies_per_system; // Offset for this galaxy

        // 1) H2D copies AoS data as in original version
        runtime_error = cudaMemcpyAsync(d_all_coords     + offset,
                                        h_data.all_coords + offset,
                                        bytes_coor_system,
                                        cudaMemcpyHostToDevice,
                                        streams[stream_id]);
        CHECK_CUDA_FAIL(runtime_error);

        runtime_error = cudaMemcpyAsync(d_all_velocities     + offset,
                                        h_data.all_velocities + offset,
                                        bytes_vel_system,
                                        cudaMemcpyHostToDevice,
                                        streams[stream_id]);
        CHECK_CUDA_FAIL(runtime_error);

        // 2) Pack AoS into SoA for this system
        pack_positions_kernel<<<grid_dims, block_dims, 0, streams[stream_id]>>>(
            d_all_coords, d_x, d_y, d_z, bodies_per_system, offset);
        runtime_error = cudaGetLastError();
        CHECK_CUDA_FAIL(runtime_error);

        pack_velocities_kernel<<<grid_dims, block_dims, 0, streams[stream_id]>>>(
            d_all_velocities, d_vx, d_vy, d_vz, bodies_per_system, offset);
        runtime_error = cudaGetLastError();
        CHECK_CUDA_FAIL(runtime_error);

        // 3) nIters time steps on SoA arrays
        for (int iter = 0; iter < nIters; iter++) {
            update_velocities<<<grid_dims, block_dims, 0, streams[stream_id]>>>(
            tex_x, tex_y, tex_z,
            d_vx, d_vy, d_vz,
            bodies_per_system,
            offset,
            dt);
            // if (use_inaccurate) {
            //     update_velocities_inaccurate<<<grid_dims, block_dims, 0, streams[stream_id]>>>(
            //         tex_x, tex_y, tex_z,            // exact fine field
            //         tex_x_coarse, tex_y_coarse, tex_z_coarse, // coarse interpolated field
            //         d_vx, d_vy, d_vz,
            //         bodies_per_system,
            //         offset,
            //         coarse_stride,
            //         coarse_size,
            //         dt);
            // } else {
            //     update_velocities<<<grid_dims, block_dims, 0, streams[stream_id]>>>(
            //         tex_x, tex_y, tex_z,
            //         d_vx, d_vy, d_vz,
            //         bodies_per_system,
            //         offset,
            //         dt);
            // }
            runtime_error = cudaGetLastError();
            CHECK_CUDA_FAIL(runtime_error);

            update_positions<<<grid_dims, block_dims, 0, streams[stream_id]>>>(
                d_x, d_y, d_z,
                d_vx, d_vy, d_vz,
                bodies_per_system,
                offset,
                dt);
            runtime_error = cudaGetLastError();
            CHECK_CUDA_FAIL(runtime_error);
        }

        // 4) Unpack SoA back into AoS d_all_coords for D2H
        unpack_positions_kernel<<<grid_dims, block_dims, 0, streams[stream_id]>>>(
            d_all_coords, d_x, d_y, d_z, bodies_per_system, offset);
        runtime_error = cudaGetLastError();
        CHECK_CUDA_FAIL(runtime_error);

        // 5) D2H copy enqueued in same stream (after kernels)
        runtime_error = cudaMemcpyAsync(h_device_out.all_coords + offset,
                                        d_all_coords           + offset,
                                        bytes_coor_system,
                                        cudaMemcpyDeviceToHost,
                                        streams[stream_id]);
        CHECK_CUDA_FAIL(runtime_error);
    }
    
    // Wait for all queued work in all streams
    cudaDeviceSynchronize();
    totalTime = GetTimer() / 1000.0;

    interactions_per_system = (double) bodies_per_system * bodies_per_system;
    total_interactions      = interactions_per_system * num_systems * nIters;

    printf("Total Time: %.3f seconds\n", totalTime);
    printf("Average Throughput: %0.3f Billion Interactions / second\n\n",
           1e-9 * total_interactions / totalTime);

    printf("Final position of System 0, Body 0: %.4f, %.4f, %.4f\n",
           h_device_out.all_coords[0].x,
           h_device_out.all_coords[0].y,
           h_device_out.all_coords[0].z);
    printf("Final position of System 0, Body 1: %.4f, %.4f, %.4f\n",
           h_device_out.all_coords[1].x,
           h_device_out.all_coords[1].y,
           h_device_out.all_coords[1].z);
    printf("----------------------------\n");
}

/* ==================== CPU COMPARISON (UNCHANGED) ==================== */

void compare_results(int total_bodies, int bodies_per_system) {
    int errors = 0;
    for (int i = 0; i < total_bodies; i++) {
        float dx = ABS(omp_out.all_coords[i].x - h_device_out.all_coords[i].x);
        float dy = ABS(omp_out.all_coords[i].y - h_device_out.all_coords[i].y);
        float dz = ABS(omp_out.all_coords[i].z - h_device_out.all_coords[i].z);

        if (dx > 0.001 || dy > 0.001 || dz > 0.001) {
            errors++;
            if (errors <= 5) {
                printf("Mismatch at Body %d (System %d):\n", i, i / bodies_per_system);
                printf("  CPU: x=%.4f, y=%.4f, z=%.4f\n", omp_out.all_coords[i].x, \
                        omp_out.all_coords[i].y, omp_out.all_coords[i].z);
                printf("  GPU: x=%.4f, y=%.4f, z=%.4f\n", h_device_out.all_coords[i].x, \
                         h_device_out.all_coords[i].y, h_device_out.all_coords[i].z);
                printf("  Diff: %.4f, %.4f, %.4f\n", dx, dy, dz);
            }
        }
    }

    if (errors > 0) {
        int success = total_bodies - errors;
        double success_rate = (100.0*success) / total_bodies;
        printf("Total Failures: %d / %d\n", errors, total_bodies);
        printf("Success Rate: %.3f%%\n", success_rate);
    } else {
        printf("Correct!\n");
    }
}

/* ==================== MAIN (CPU LOGIC UNCHANGED, + SoA + TEX) ==================== */

int main(const int argc, const char *argv[]) {
    /* Default Configuration */
    int num_systems = 32;           /* Number of independent galaxies */
    int bodies_per_system = 8192;   /* Number of bodies per galaxy */
    int nIters = 20;                /* Simulation steps */ 
    
    const float dt = 0.01f;
    FILE *fp;
    int total_bodies;

    cudaError_t error_coords, error_velocs;
    int bytes_coords, bytes_veloc;

    // Check environment variable to switch between accurate and inaccurate
    char *inaccurate_env = getenv("INACCURATE");
    if (inaccurate_env && atoi(inaccurate_env) == 1) {
        use_inaccurate = 1;
        printf("Using INACCURATE kernel (coarse texture approximation)\n");
    } else {
        use_inaccurate = 0;
        printf("Using ACCURATE kernel (exact point sampling)\n");
    }

    // create streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    /* Attempt to load dataset */
    fp = fopen("galaxy_data.bin", "rb");
    if (fp) {
        fread(&num_systems, sizeof(int), 1, fp);
        fread(&bodies_per_system, sizeof(int), 1, fp);
        printf("Found dataset: %d systems of %d bodies.\n", num_systems,
                bodies_per_system);
    } else {
        printf("No dataset found. Using random initialization.\n");
    }

    /* Allocate memory for ALL systems */
    total_bodies = num_systems * bodies_per_system;
    bytes_coords = total_bodies * sizeof(coords_T);
    bytes_veloc = total_bodies * sizeof(velocity_T);

    cudaError_t error_h_data_coor = cudaHostAlloc((coords_T **) &h_data.all_coords, bytes_coords, cudaHostAllocMapped);
    cudaError_t error_h_data_vel = cudaHostAlloc((velocity_T **) &h_data.all_velocities, bytes_veloc, cudaHostAllocMapped);
    omp_out.all_coords = (coords_T *) malloc(bytes_coords);
    omp_out.all_velocities = (velocity_T *) malloc(bytes_veloc);
    cudaError_t error_dev_out_coor = cudaHostAlloc((coords_T **) &h_device_out.all_coords, bytes_coords, cudaHostAllocMapped);
    h_device_out.all_velocities = NULL; // only position needed for the output
    if(!IS_CUDA_MALLOC_OK(error_h_data_coor) || !IS_CUDA_MALLOC_OK(error_h_data_vel) || \
        IS_NULL(omp_out.all_coords) || IS_NULL(omp_out.all_velocities) || \
        !IS_CUDA_MALLOC_OK(error_dev_out_coor)) {
        printf("Host: Memory allocation failed!\n");
        cudaFreeHost(h_data.all_coords);
        cudaFreeHost(h_data.all_velocities);   
        FREE_MEM(omp_out.all_coords);
        FREE_MEM(omp_out.all_velocities);
        cudaFreeHost(h_device_out.all_coords);
        return 1;
    }

    /* Initialize data */
    if (fp) {
        for (int i = 0; i < total_bodies; i++) { 
            fread(&(h_data.all_coords[i]), sizeof(coords_T), 1, fp);
            // skip velocities
            fseek(fp, sizeof(velocity_T), SEEK_CUR);
        }    
        // set to start and skip 2 first integers
        fseek(fp, 2*sizeof(int), SEEK_SET);
        for (int i = 0; i < total_bodies; i++) {
            // skip positions
            fseek(fp, sizeof(coords_T), SEEK_CUR); 
            fread(&(h_data.all_velocities[i]), sizeof(velocity_T), 1, fp);
        }
        fclose(fp);
    }
    else {
        printf("galaxy_data.bin is missing!\n");
    }

    // Save h_data for transferring to device
    memcpy(omp_out.all_coords, h_data.all_coords, bytes_coords);   
    memcpy(omp_out.all_velocities, h_data.all_velocities, bytes_veloc);

    // overwrite during run time if you want
    char *d = getenv("DEBUG");
    if (d) debug = atoi(d);

    // Run parallel CPU
    if (debug)
        run_omp(num_systems, nIters, bodies_per_system, dt);
    
    // allocate memory on device (AoS, as before)
    error_coords = cudaMalloc((void **)&d_data.all_coords, bytes_coords);
    error_velocs = cudaMalloc((void **)&d_data.all_velocities, bytes_veloc);
    if(!IS_CUDA_MALLOC_OK(error_coords) || !IS_CUDA_MALLOC_OK(error_velocs)) {
        printf("CUDA: Memory allocation Failed!\n");
        cudaFreeHost(h_data.all_coords);
        cudaFreeHost(h_data.all_velocities);   
        free(omp_out.all_coords);
        free(omp_out.all_velocities);
        cudaFreeHost(h_device_out.all_coords);
        if (IS_CUDA_MALLOC_OK(error_coords)) cudaFree(d_data.all_coords);
        if (IS_CUDA_MALLOC_OK(error_velocs)) cudaFree(d_data.all_velocities);
        cudaDeviceReset();
        return 1;
    }

    // allocate SoA arrays for GPU
    cudaError_t err;
    err = cudaMalloc((void**)&d_x,  total_bodies * sizeof(float)); CHECK_CUDA_FAIL(err);
    err = cudaMalloc((void**)&d_y,  total_bodies * sizeof(float)); CHECK_CUDA_FAIL(err);
    err = cudaMalloc((void**)&d_z,  total_bodies * sizeof(float)); CHECK_CUDA_FAIL(err);
    err = cudaMalloc((void**)&d_vx, total_bodies * sizeof(float)); CHECK_CUDA_FAIL(err);
    err = cudaMalloc((void**)&d_vy, total_bodies * sizeof(float)); CHECK_CUDA_FAIL(err);
    err = cudaMalloc((void**)&d_vz, total_bodies * sizeof(float)); CHECK_CUDA_FAIL(err);

    // --- Create texture objects for x, y, z (1D linear, point sampling = ACCURATE) ---
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeClamp;  // no wrap
    texDesc.filterMode       = cudaFilterModePoint;   // no interpolation
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;                     // use integer indices

    // x
    resDesc.res.linear.devPtr      = d_x;
    resDesc.res.linear.desc        = cudaCreateChannelDesc<float>();
    resDesc.res.linear.sizeInBytes = total_bodies * sizeof(float);
    err = cudaCreateTextureObject(&tex_x, &resDesc, &texDesc, NULL);
    CHECK_CUDA_FAIL(err);

    // y
    resDesc.res.linear.devPtr      = d_y;
    resDesc.res.linear.desc        = cudaCreateChannelDesc<float>();
    resDesc.res.linear.sizeInBytes = total_bodies * sizeof(float);
    err = cudaCreateTextureObject(&tex_y, &resDesc, &texDesc, NULL);
    CHECK_CUDA_FAIL(err);

    // z
    resDesc.res.linear.devPtr      = d_z;
    resDesc.res.linear.desc        = cudaCreateChannelDesc<float>();
    resDesc.res.linear.sizeInBytes = total_bodies * sizeof(float);
    err = cudaCreateTextureObject(&tex_z, &resDesc, &texDesc, NULL);
    CHECK_CUDA_FAIL(err);

    // ---------- Inaccurate (coarse) field setup ----------
    coarse_stride = 4;  // tune this
    coarse_size   = (total_bodies + coarse_stride - 1) / coarse_stride;

    err = cudaMalloc((void**)&d_x_coarse, coarse_size * sizeof(float)); CHECK_CUDA_FAIL(err);
    err = cudaMalloc((void**)&d_y_coarse, coarse_size * sizeof(float)); CHECK_CUDA_FAIL(err);
    err = cudaMalloc((void**)&d_z_coarse, coarse_size * sizeof(float)); CHECK_CUDA_FAIL(err);

    // Build the coarse field once (from initial positions).
    // Positions will change over time, so for "less wrong" results you might rebuild occasionally.
    int threads = 256;
    int blocks  = (coarse_size + threads - 1) / threads;
    build_coarse_field<<<blocks, threads>>>(d_x, d_y, d_z,
                                            d_x_coarse, d_y_coarse, d_z_coarse,
                                            total_bodies, coarse_stride);
    err = cudaGetLastError();
    CHECK_CUDA_FAIL(err);

    // Create linear-filtered textures for coarse field
    cudaResourceDesc resDescC;
    memset(&resDescC, 0, sizeof(resDescC));
    resDescC.resType = cudaResourceTypeLinear;

    cudaTextureDesc texDescC;
    memset(&texDescC, 0, sizeof(texDescC));
    texDescC.addressMode[0]   = cudaAddressModeClamp;
    texDescC.filterMode       = cudaFilterModeLinear;     // <-- interpolation!
    texDescC.readMode         = cudaReadModeElementType;
    texDescC.normalizedCoords = 0;

    // x_coarse
    resDescC.res.linear.devPtr      = d_x_coarse;
    resDescC.res.linear.desc        = cudaCreateChannelDesc<float>();
    resDescC.res.linear.sizeInBytes = coarse_size * sizeof(float);
    err = cudaCreateTextureObject(&tex_x_coarse, &resDescC, &texDescC, NULL);
    CHECK_CUDA_FAIL(err);

    // y_coarse
    resDescC.res.linear.devPtr      = d_y_coarse;
    resDescC.res.linear.desc        = cudaCreateChannelDesc<float>();
    resDescC.res.linear.sizeInBytes = coarse_size * sizeof(float);
    err = cudaCreateTextureObject(&tex_y_coarse, &resDescC, &texDescC, NULL);
    CHECK_CUDA_FAIL(err);

    // z_coarse
    resDescC.res.linear.devPtr      = d_z_coarse;
    resDescC.res.linear.desc        = cudaCreateChannelDesc<float>();
    resDescC.res.linear.sizeInBytes = coarse_size * sizeof(float);
    err = cudaCreateTextureObject(&tex_z_coarse, &resDescC, &texDescC, NULL);
    CHECK_CUDA_FAIL(err);

    // Run on GPU
    run_device(d_data.all_coords, d_data.all_velocities ,num_systems, bytes_coords,\
                bytes_veloc, nIters, bodies_per_system, dt);

    /* COMPARE RESULTS */
    if(debug)
        compare_results(total_bodies, bodies_per_system);

    cudaFreeHost(h_data.all_coords);
    cudaFreeHost(h_data.all_velocities);   
    free(omp_out.all_coords);
    free(omp_out.all_velocities);
    cudaFreeHost(h_device_out.all_coords);
    cudaFree(d_data.all_coords);
    cudaFree(d_data.all_velocities);    

    // destroy textures and free SoA
    cudaDestroyTextureObject(tex_x);
    cudaDestroyTextureObject(tex_y);
    cudaDestroyTextureObject(tex_z);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);

    // destroy coarse textures and free coarse arrays
    cudaDestroyTextureObject(tex_x_coarse);
    cudaDestroyTextureObject(tex_y_coarse);
    cudaDestroyTextureObject(tex_z_coarse);
    cudaFree(d_x_coarse); cudaFree(d_y_coarse); cudaFree(d_z_coarse);
    cudaDeviceReset();
    return 0;
}
