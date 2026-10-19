#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../timer.h"

#define SOFTENING 0.01f 

// Optimized constants
#define THREADS_PER_BLOCK 256
#define TILE_SIZE 256
#define NUM_STREAMS 16

#define ABS(val)    ((val)<0.0 ? (-(val)) : (val))
#define IS_NULL(ptr) (((ptr) == (NULL)) ? 1 : 0)
// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#ifdef DEBUG
    int debug = 1;
#else
    int debug = 0;
#endif

// CPU structures (AoS - unchanged)
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

// GPU Structure-of-Arrays (SoA)
typedef struct {
    float *x, *y, *z;
    float *vx, *vy, *vz;
} Body_SoA;

// Per-device context for multi-GPU
typedef struct {
    Body_SoA d_bodies;
    coords_T *h_out_coords;
    cudaStream_t streams[NUM_STREAMS];
    int num_systems_local;
    int sys_start_global;
} DeviceContext;

Universe h_data, h_device_out, omp_out;

/* CPU CODE */
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
    printf("Total Time: %.3f seconds\n", totalTime);
    printf("Average Throughput CPU: %0.3f Billion Interactions / second\n\n",
           1e-9 * total_interactions / totalTime);

    /* Dump final state of System 0, Body 0 and 1 for verification comparison */
    printf("Final position of System 0, Body 0: %.4f, %.4f, %.4f\n",
           omp_out.all_coords[0].x, omp_out.all_coords[0].y, omp_out.all_coords[0].z);
    printf("Final position of System 0, Body 1: %.4f, %.4f, %.4f\n",
           omp_out.all_coords[1].x, omp_out.all_coords[1].y, omp_out.all_coords[1].z);
    printf("----------------------------\n");
}

/* GPU KERNELS */
__global__ void update_velocities_soa_tiled(const float* __restrict__ x, const float* __restrict__ y, \
                                            const float* __restrict__ z, float* __restrict__ vx, \
                                            float* __restrict__ vy, float* __restrict__ vz, \
                                            int bodies_per_system, float dt, int device_id) {
    int sys = blockIdx.z;         
    int pointer_index = sys * bodies_per_system;
    
    // Body index within this system
    int body_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (body_id >= bodies_per_system) return;

    // Shared memory for tiling
    __shared__ float shared_x[TILE_SIZE];
    __shared__ float shared_y[TILE_SIZE];
    __shared__ float shared_z[TILE_SIZE];

    // Load this body's position
    float body_x = x[pointer_index + body_id];
    float body_y = y[pointer_index + body_id];
    float body_z = z[pointer_index + body_id];

    // Accumulate forces
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    int num_tiles = (bodies_per_system + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over tiles
    #pragma unroll 4
    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_idx = tile * TILE_SIZE + threadIdx.x;
        
        // Cooperatively load tile into shared memory
        if (tile_idx < bodies_per_system) {
            shared_x[threadIdx.x] = x[pointer_index + tile_idx];
            shared_y[threadIdx.x] = y[pointer_index + tile_idx];
            shared_z[threadIdx.x] = z[pointer_index + tile_idx];
        }
        __syncthreads();

        // Compute interactions with all bodies in this tile
        #pragma unroll 8
        for (int j = 0; j < TILE_SIZE; j++) {
            int idx = tile * TILE_SIZE + j;
            if (idx >= bodies_per_system) break;

            float dx = shared_x[j] - body_x;
            float dy = shared_y[j] - body_y;
            float dz = shared_z[j] - body_z;

            float distSqr = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, SOFTENING)));
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            fx = fmaf(dx, invDist3, fx);
            fy = fmaf(dy, invDist3, fy);
            fz = fmaf(dz, invDist3, fz);
        }
        __syncthreads();
    }

    // Update velocities
    vx[pointer_index + body_id] += dt * fx;
    vy[pointer_index + body_id] += dt * fy;
    vz[pointer_index + body_id] += dt * fz;
}

__global__ void update_positions_soa(float* __restrict__ x, float* __restrict__ y, \
                                     float* __restrict__ z, const float* __restrict__ vx, \
                                     const float* __restrict__ vy, const float* __restrict__ vz, \
                                     int bodies_per_system, float dt, int device_id) {
    int sys = blockIdx.y;  
    int pointer_index = sys * bodies_per_system;
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bodies_per_system) return;

    int idx = pointer_index + i;
    x[idx] += vx[idx] * dt;
    y[idx] += vy[idx] * dt;
    z[idx] += vz[idx] * dt;
}

/* MULTI-GPU RUNNER */
void run_device_multi_gpu(int num_systems, int bodies_per_system, int nIters, float dt) {
    double totalTime;
    int total_bodies = num_systems * bodies_per_system;
    int bytes_coords = total_bodies * sizeof(coords_T);
    int bytes_vels = total_bodies * sizeof(velocity_T);

    // Detect GPUs
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count == 0) {
        printf("No CUDA devices found.\n");
        exit(1);
    }

    printf("\n----------------------------\n");
    printf("Using %d GPU(s) for simulation\n", dev_count);

    // Allocate per-device contexts
    DeviceContext *dev_ctx = (DeviceContext*)malloc(dev_count * sizeof(DeviceContext));

    // Partition systems across GPUs
    int base_sys_per_dev = num_systems / dev_count;
    int extra = num_systems % dev_count;

    int sys_start = 0;
    for (int dev = 0; dev < dev_count; dev++) {
        CUDA_CHECK(cudaSetDevice(dev));

        int nsys = base_sys_per_dev + (dev < extra ? 1 : 0);
        dev_ctx[dev].num_systems_local = nsys;
        dev_ctx[dev].sys_start_global = sys_start;

        int bodies_local = nsys * bodies_per_system;

        if (nsys > 0) {
            // Allocate SoA device memory
            CUDA_CHECK(cudaMalloc((void**)&dev_ctx[dev].d_bodies.x,  bodies_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc((void**)&dev_ctx[dev].d_bodies.y,  bodies_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc((void**)&dev_ctx[dev].d_bodies.z,  bodies_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc((void**)&dev_ctx[dev].d_bodies.vx, bodies_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc((void**)&dev_ctx[dev].d_bodies.vy, bodies_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc((void**)&dev_ctx[dev].d_bodies.vz, bodies_local * sizeof(float)));

            // Allocate pinned host output buffer
            CUDA_CHECK(cudaHostAlloc((void**)&dev_ctx[dev].h_out_coords,
                                     bodies_local * sizeof(coords_T),
                                     cudaHostAllocMapped));

            // Create streams
            for (int s = 0; s < NUM_STREAMS; s++) {
                CUDA_CHECK(cudaStreamCreate(&dev_ctx[dev].streams[s]));
            }
        } else {
            dev_ctx[dev].d_bodies.x = NULL;
            dev_ctx[dev].h_out_coords = NULL;
        }

        sys_start += nsys;
    }

    // Temporary host SoA buffers for packing
    Body_SoA h_soa;
    h_soa.x  = (float*)malloc(total_bodies * sizeof(float));
    h_soa.y  = (float*)malloc(total_bodies * sizeof(float));
    h_soa.z  = (float*)malloc(total_bodies * sizeof(float));
    h_soa.vx = (float*)malloc(total_bodies * sizeof(float));
    h_soa.vy = (float*)malloc(total_bodies * sizeof(float));
    h_soa.vz = (float*)malloc(total_bodies * sizeof(float));

    // Pack AoS to SoA on host
    for (int i = 0; i < total_bodies; i++) {
        h_soa.x[i]  = h_data.all_coords[i].x;
        h_soa.y[i]  = h_data.all_coords[i].y;
        h_soa.z[i]  = h_data.all_coords[i].z;
        h_soa.vx[i] = h_data.all_velocities[i].vx;
        h_soa.vy[i] = h_data.all_velocities[i].vy;
        h_soa.vz[i] = h_data.all_velocities[i].vz;
    }

    // H2D: Copy initial data to each GPU
    for (int dev = 0; dev < dev_count; dev++) {
        if (dev_ctx[dev].num_systems_local == 0) continue;

        CUDA_CHECK(cudaSetDevice(dev));

        int sys_start_global = dev_ctx[dev].sys_start_global;
        int num_systems_local = dev_ctx[dev].num_systems_local;
        int body_offset = sys_start_global * bodies_per_system;
        int body_count = num_systems_local * bodies_per_system;

        CUDA_CHECK(cudaMemcpy(dev_ctx[dev].d_bodies.x,  h_soa.x  + body_offset, body_count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_ctx[dev].d_bodies.y,  h_soa.y  + body_offset, body_count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_ctx[dev].d_bodies.z,  h_soa.z  + body_offset, body_count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_ctx[dev].d_bodies.vx, h_soa.vx + body_offset, body_count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_ctx[dev].d_bodies.vy, h_soa.vy + body_offset, body_count * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_ctx[dev].d_bodies.vz, h_soa.vz + body_offset, body_count * sizeof(float), cudaMemcpyHostToDevice));
    }

    // computation
    StartTimer();

    for (int dev = 0; dev < dev_count; dev++) {
        if (dev_ctx[dev].num_systems_local == 0) continue;
        
        CUDA_CHECK(cudaSetDevice(dev));
        
        int num_systems_local = dev_ctx[dev].num_systems_local;
        cudaStream_t sim_stream = dev_ctx[dev].streams[0];
        
        // Calculate grid dimensions ONCE per device
        int blocks_per_sys = (bodies_per_system + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        dim3 grid_vel(blocks_per_sys, 1, num_systems_local);
        dim3 grid_pos(blocks_per_sys, num_systems_local);
        dim3 block(THREADS_PER_BLOCK);
        
        for (int iter = 0; iter < nIters; iter++) {
            update_velocities_soa_tiled<<<grid_vel, block, 0, sim_stream>>>(
                dev_ctx[dev].d_bodies.x, dev_ctx[dev].d_bodies.y, dev_ctx[dev].d_bodies.z,
                dev_ctx[dev].d_bodies.vx, dev_ctx[dev].d_bodies.vy, dev_ctx[dev].d_bodies.vz,
                bodies_per_system, dt, 0);
            CUDA_CHECK(cudaGetLastError());

            update_positions_soa<<<grid_pos, block, 0, sim_stream>>>(
                dev_ctx[dev].d_bodies.x, dev_ctx[dev].d_bodies.y, dev_ctx[dev].d_bodies.z,
                dev_ctx[dev].d_bodies.vx, dev_ctx[dev].d_bodies.vy, dev_ctx[dev].d_bodies.vz,
                bodies_per_system, dt, 0);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // Synchronize all GPUs
    for (int dev = 0; dev < dev_count; dev++) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    totalTime = GetTimer() / 1000.0;


    // Copy results back to host
    for (int dev = 0; dev < dev_count; dev++) {
        if (dev_ctx[dev].num_systems_local == 0) continue;

        CUDA_CHECK(cudaSetDevice(dev));

        int sys_start_global = dev_ctx[dev].sys_start_global;
        int num_systems_local = dev_ctx[dev].num_systems_local;
        int body_offset = sys_start_global * bodies_per_system;
        int body_count = num_systems_local * bodies_per_system;

        CUDA_CHECK(cudaMemcpy(h_soa.x  + body_offset, dev_ctx[dev].d_bodies.x,  body_count * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_soa.y  + body_offset, dev_ctx[dev].d_bodies.y,  body_count * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_soa.z  + body_offset, dev_ctx[dev].d_bodies.z,  body_count * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Unpack SoA -> AoS
    for (int i = 0; i < total_bodies; i++) {
        h_device_out.all_coords[i].x = h_soa.x[i];
        h_device_out.all_coords[i].y = h_soa.y[i];
        h_device_out.all_coords[i].z = h_soa.z[i];
    }

    // Cleanup
    for (int dev = 0; dev < dev_count; dev++) {
        if (dev_ctx[dev].num_systems_local == 0) continue;

        CUDA_CHECK(cudaSetDevice(dev));
        for (int s = 0; s < NUM_STREAMS; s++) {
            CUDA_CHECK(cudaStreamDestroy(dev_ctx[dev].streams[s]));
        }
        CUDA_CHECK(cudaFree(dev_ctx[dev].d_bodies.x));
        CUDA_CHECK(cudaFree(dev_ctx[dev].d_bodies.y));
        CUDA_CHECK(cudaFree(dev_ctx[dev].d_bodies.z));
        CUDA_CHECK(cudaFree(dev_ctx[dev].d_bodies.vx));
        CUDA_CHECK(cudaFree(dev_ctx[dev].d_bodies.vy));
        CUDA_CHECK(cudaFree(dev_ctx[dev].d_bodies.vz));
        CUDA_CHECK(cudaFreeHost(dev_ctx[dev].h_out_coords));
    }

    free(dev_ctx);
    free(h_soa.x); free(h_soa.y); free(h_soa.z);
    free(h_soa.vx); free(h_soa.vy); free(h_soa.vz);

    // Metrics
    double interactions_per_system = (double)bodies_per_system * bodies_per_system;
    double total_interactions = interactions_per_system * num_systems * nIters;

    printf("Multi-GPU run on %d device(s)\n", dev_count);
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

void compare_results(int total_bodies, int bodies_per_system) {
    int errors = 0;
    for (int i = 0; i < total_bodies; i++) {
        float dx = ABS(omp_out.all_coords[i].x - h_device_out.all_coords[i].x);
        float dy = ABS(omp_out.all_coords[i].y - h_device_out.all_coords[i].y);
        float dz = ABS(omp_out.all_coords[i].z - h_device_out.all_coords[i].z);
        const float eps = 0.001f;
        if (dx > eps || dy > eps || dz > eps) {
            errors++;
            if (errors <= 5) {
                printf("Mismatch at Body %d (System %d):\n", i, i / bodies_per_system);
                printf("  CPU: x=%.4f, y=%.4f, z=%.4f\n",
                       omp_out.all_coords[i].x, omp_out.all_coords[i].y, omp_out.all_coords[i].z);
                printf("  GPU: x=%.4f, y=%.4f, z=%.4f\n",
                       h_device_out.all_coords[i].x, h_device_out.all_coords[i].y, h_device_out.all_coords[i].z);
                printf("  Diff: %.4f, %.4f, %.4f\n", dx, dy, dz);
            }
        }
    }
    if (errors == 0) {
        printf(" Correct\n");
    } else {
        printf(" Found %d mismatches\n", errors);
    }
}

int main(const int argc, const char *argv[]) {
    int num_systems = 32;
    int bodies_per_system = 8192;
    int nIters = 20;
    const float dt = 0.01f;
    FILE *fp;
    int total_bodies;

    // Load dataset
    fp = fopen("galaxy_data.bin", "rb");
    if (fp) {
        fread(&num_systems, sizeof(int), 1, fp);
        fread(&bodies_per_system, sizeof(int), 1, fp);
        printf("Found dataset: %d systems of %d bodies.\n", num_systems, bodies_per_system);
    } else {
        printf("No dataset found. Using random initialization.\n");
    }

    // Allocate memory
    total_bodies = num_systems * bodies_per_system;
    int bytes_coords = total_bodies * sizeof(coords_T);
    int bytes_vels = total_bodies * sizeof(velocity_T);

    CUDA_CHECK(cudaHostAlloc((void**)&h_data.all_coords, bytes_coords, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc((void**)&h_data.all_velocities, bytes_vels, cudaHostAllocMapped));
    omp_out.all_coords = (coords_T*)malloc(bytes_coords);
    omp_out.all_velocities = (velocity_T*)malloc(bytes_vels);
    CUDA_CHECK(cudaHostAlloc((void**)&h_device_out.all_coords, bytes_coords, cudaHostAllocMapped));
    h_device_out.all_velocities = NULL;

    if (IS_NULL(omp_out.all_coords) || IS_NULL(omp_out.all_velocities)) {
        printf("Host: Memory allocation failed!\n");
        return 1;
    }

    // Initialize data
    if (fp) {
        for (int i = 0; i < total_bodies; i++) { 
            fread(&(h_data.all_coords[i]), sizeof(coords_T), 1, fp);
            fseek(fp, sizeof(velocity_T), SEEK_CUR);
        }    
        fseek(fp, 2*sizeof(int), SEEK_SET);
        for (int i = 0; i < total_bodies; i++) {
            fseek(fp, sizeof(coords_T), SEEK_CUR); 
            fread(&(h_data.all_velocities[i]), sizeof(velocity_T), 1, fp);
        }
        fclose(fp);
    } else {
        printf("galaxy_data.bin is missing!\n");
        return 1;
    }

    // Save copy for CPU
    memcpy(omp_out.all_coords, h_data.all_coords, bytes_coords);   
    memcpy(omp_out.all_velocities, h_data.all_velocities, bytes_vels);

    // Check debug flag
    char *d = getenv("DEBUG");
    if (d) debug = atoi(d);

    // Run CPU (if debug)
    if (debug)
        run_omp(num_systems, nIters, bodies_per_system, dt);
    
    // Run GPU
    run_device_multi_gpu(num_systems, bodies_per_system, nIters, dt);

    // Compare results
    if (debug)
        compare_results(total_bodies, bodies_per_system);

    // Cleanup
    cudaFreeHost(h_data.all_coords);
    cudaFreeHost(h_data.all_velocities);   
    free(omp_out.all_coords);
    free(omp_out.all_velocities);
    cudaFreeHost(h_device_out.all_coords);
    
    cudaDeviceReset();
    return 0;
}