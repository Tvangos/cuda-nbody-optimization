#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../timer.h"

#define SOFTENING 0.01f 

// 1024 threads per block
#define THREADS_PER_BLOCK 1024
// Bodies per tile
#define TILE_SIZE 512
#define NUM_STREAMS 16
#define BLOCKS_PER_SYS 32

#define ABS(val)    ((val)<0.0 ? (-(val)) : (val))

#define IS_NULL(ptr) (((ptr) == (NULL)) ? 1 : 0)
#define IS_CUDA_MALLOC_OK(err) (((err) == (cudaSuccess)) ? (1) : (0))
#define FREE_MEM(ptr) (((ptr) != (NULL)) ? (free(ptr)) : ((void)(ptr))) // Free CPU mem
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

__global__ void update_velocities(const coords_T *__restrict__ coords_sys,
                                  velocity_T *__restrict__ velocs_sys,
                                  int bodies_per_system,
                                  float dt) {
    __shared__ coords_T shared_coords[TILE_SIZE];

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;

    const int tiles_per_sys = (bodies_per_system + TILE_SIZE - 1) / TILE_SIZE;

    for (int i = tid; i < bodies_per_system; i += stride) {
        
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
        coords_T curr = coords_sys[i];

        for (int tile = 0; tile < tiles_per_sys; tile++) {
            int tile_offset = tile * TILE_SIZE;

            // Load tile cooperatively using read-only cache
            if (threadIdx.x < TILE_SIZE) {
                int load_idx = tile_offset + threadIdx.x;
                if (load_idx < bodies_per_system) {
                    const float *base = reinterpret_cast<const float*>(coords_sys);
                    const float *src  = base + 3 * load_idx;  // x,y,z

                    coords_T tmp;
                    tmp.x = __ldg(src + 0);
                    tmp.y = __ldg(src + 1);
                    tmp.z = __ldg(src + 2);
                    shared_coords[threadIdx.x] = tmp;
                }
            }
            __syncthreads();

            int tile_end = min(TILE_SIZE, bodies_per_system - tile_offset);

            #pragma unroll 8
            for (int j = 0; j < tile_end; j++) {
                float dx = shared_coords[j].x - curr.x;
                float dy = shared_coords[j].y - curr.y;
                float dz = shared_coords[j].z - curr.z;
                
                float distSqr = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, SOFTENING)));
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
            __syncthreads();
        }

        velocs_sys[i].vx += dt * Fx;
        velocs_sys[i].vy += dt * Fy;
        velocs_sys[i].vz += dt * Fz;
    }
}

__global__ void update_positions(coords_T *__restrict__ coords_sys,
                                 const velocity_T *__restrict__ velocs_sys,
                                 int bodies_per_system,
                                 float dt) 
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;

    for (int i = tid; i < bodies_per_system; i += stride) {
        coords_sys[i].x += velocs_sys[i].vx * dt;
        coords_sys[i].y += velocs_sys[i].vy * dt;
        coords_sys[i].z += velocs_sys[i].vz * dt;
    }
}


void run_device(coords_T *d_all_coords, velocity_T *d_all_velocities,
                int num_systems, int bytes_coords, int bytes_velocs,
                int nIters, int bodies_per_system, float dt) {

    double totalTime;
    cudaError_t runtime_error;
    double interactions_per_system, total_interactions;

    printf("\n----------------------------\n");
    printf("Running Device simulation for %d systems...\n", num_systems);

    int bytes_coor_system = sizeof(coords_T)   * bodies_per_system;
    int bytes_vel_system  = sizeof(velocity_T) * bodies_per_system;

    dim3 grid_dims  (BLOCKS_PER_SYS, 1);
    dim3 block_dims (THREADS_PER_BLOCK, 1);
    
    StartTimer();
    for (int sys = 0; sys < num_systems; sys++) {
        int stream_id = sys % NUM_STREAMS;       
        int offset    = sys * bodies_per_system; 
        
        // Copy data on device
        runtime_error = cudaMemcpyAsync((coords_T *)(d_all_coords+offset), \
                                        (coords_T *)(h_data.all_coords+offset), \
                                        bytes_coor_system, cudaMemcpyHostToDevice, streams[stream_id]);
        CHECK_CUDA_FAIL(runtime_error);
        runtime_error = cudaMemcpyAsync((velocity_T *)(d_all_velocities+offset), \
                                        (velocity_T *)(h_data.all_velocities+offset), \
                                        bytes_vel_system, cudaMemcpyHostToDevice, streams[stream_id]);
        CHECK_CUDA_FAIL(runtime_error);

        for (int iter = 0; iter < nIters; iter++) {
            update_velocities<<<grid_dims, block_dims, 0, streams[stream_id]>>>
                            ((d_all_coords + offset),(d_all_velocities + offset), bodies_per_system, dt);
            runtime_error = cudaGetLastError();
            CHECK_CUDA_FAIL(runtime_error);

            update_positions<<<grid_dims, block_dims, 0, streams[stream_id]>>>
                            ((d_all_coords + offset), (d_all_velocities + offset), bodies_per_system, dt);
            runtime_error = cudaGetLastError();
            CHECK_CUDA_FAIL(runtime_error);
        }
       
        // Copy device output back on host
        runtime_error = cudaMemcpyAsync((coords_T *)(h_device_out.all_coords + offset),(coords_T *)(d_all_coords + offset), \
                            bytes_coor_system, cudaMemcpyDeviceToHost, streams[stream_id]);
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
        double success_rate = (100*success) / total_bodies;
        printf("Total Failures: %d / %d\n", errors, total_bodies);
        printf("Success Rate: %.3f%\n", success_rate);
    } else {
        printf("Correct!\n");
    }
}

int main(const int argc, const char *argv[]) {
    /* Default Configuration */
    int num_systems = 32;       	/* Number of independent galaxies */
    int bodies_per_system = 8192;	/* Number of bodies per galaxy */
    int nIters = 20;            	/* Simulation steps */ 
    
    const float dt = 0.01f;
    FILE *fp;
    int total_bodies;
    float *buf;

    cudaError_t error_coords, error_velocs;
    int bytes_coords, bytes_veloc;

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
    // no random init 
    else {
        printf("galaxy_data.bin is missing!\n");
    }

    // Save h_data for transfering to device
    memcpy(omp_out.all_coords, h_data.all_coords, bytes_coords);   
    memcpy(omp_out.all_velocities, h_data.all_velocities, bytes_veloc);

    // overwrite during run time if you want
    char *d = getenv("DEBUG");
    if (d) debug = atoi(d);

    // Run parallel CPU
    if (debug)
        run_omp(num_systems, nIters, bodies_per_system, dt);
    
    // allocate memory on device
    error_coords = cudaMalloc((void **)&d_data.all_coords, bytes_coords);
    error_velocs = cudaMalloc((void **)&d_data.all_velocities, bytes_veloc);
    if(!IS_CUDA_MALLOC_OK(error_coords) || !IS_CUDA_MALLOC_OK(error_velocs)) {
        printf("CUDA: Memory allocation Failed!\n");
        cudaFreeHost(h_data.all_coords);
        cudaFreeHost(h_data.all_velocities);   
        free(omp_out.all_coords);
        free(omp_out.all_velocities);
        cudaFreeHost(h_device_out.all_coords);
        IS_CUDA_MALLOC_OK(error_coords) ? cudaFree(d_data.all_coords) : 0;
        IS_CUDA_MALLOC_OK(error_velocs) ? cudaFree(d_data.all_velocities) : 0;
        cudaDeviceReset();
        return 1;
    }
   
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
    
    cudaDeviceReset();
    return 0;
}
