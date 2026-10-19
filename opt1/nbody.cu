#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../timer.h"

#define SOFTENING 0.01f 

// 1024 threads per block
#define THREADS_PER_BLOCK 1024

#define ABS(val)    ((val)<0.0 ? (-(val)) : (val))

#define IS_NULL(ptr) (((ptr) == (NULL)) ? 1 : 0)
#define IS_CUDA_MALLOC_OK(err) (((err) == (cudaSuccess)) ? (1) : (0))
#define FREE_MEM(ptr) (((ptr) != (NULL)) ? (free(ptr)) : ((void)(ptr))) // Free CPU mem
// Check for run time error and free allocated memory if you detect failure
#define CHECK_CUDA_FAIL(err) \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s (%d) in %s, line %i\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
        FREE_MEM(h_data.all_coords); \
        FREE_MEM(h_data.all_velocities); \
        FREE_MEM(omp_out.all_coords); \
        FREE_MEM(omp_out.all_velocities); \
        FREE_MEM(h_device_out.all_coords); \
        FREE_MEM(h_device_out.all_velocities); \
        cudaFree(d_data.all_coords); \
        cudaFree(d_data.all_velocities); \
        cudaDeviceReset(); \
        exit(1); \
    }

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
	        dx = coor_p[j].x - curr.x;
            dy = coor_p[j].y - curr.y;
            dz = coor_p[j].z - curr.z;
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

/* Update a single body in a galaxy. Parameters:
    - array of bodies
    - time step
    - number of bodies
    - body index in data array
*/
__device__ void device_bodyForce(coords_T * coor_p, velocity_T * vel_p, \
                                 float dt, int n, int bodyIdx) {
    int j;
    float Fx, Fy, Fz, dx, dy, dz, distSqr, invDist, invDist3;

    Fx = 0.0f;
    Fy = 0.0f;
    Fz = 0.0f;
    
    coords_T curr = coor_p[bodyIdx]; 
    for (j = 0; j < n; j++) {
	    dx = coor_p[j].x - curr.x;
        dy = coor_p[j].y - curr.y;
        dz = coor_p[j].z - curr.z;
        distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        invDist = 1.0f / sqrtf(distSqr);
        invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    vel_p[bodyIdx].vx += dt * Fx;
    vel_p[bodyIdx].vy += dt * Fy;
    vel_p[bodyIdx].vz += dt * Fz;
}

/* Integrate position of single body.
    - array of bodies
    - time step
    - body index in data array
*/
__device__ void device_integrate(coords_T * coor_p, velocity_T * vel_p, \
                                float dt, int bodyIdx) {

    coor_p[bodyIdx].x += vel_p[bodyIdx].vx * dt;
    coor_p[bodyIdx].y += vel_p[bodyIdx].vy * dt;
    coor_p[bodyIdx].z += vel_p[bodyIdx].vz * dt;
}

// kernel
__global__ void device_nbody(coords_T *d_all_coords, velocity_T * d_all_velocs, int nIters, \
                             int bodies_per_system, float dt) {
    int iter, i;
    int sys = blockIdx.x; 
    int stride = blockDim.x;

    /* Calculate offset for the galaxy */
    coords_T *coords_ptr = &(d_all_coords[sys * bodies_per_system]);
    velocity_T *veloc_ptr = &(d_all_velocs[sys * bodies_per_system]);    
    
    /* Time-steps */        
    for (iter = 1; iter <= nIters; iter++) {
        // 1 thread for bodies_per_system / THREADS_PER_BLOCK bodies
        for (i = threadIdx.x; i < bodies_per_system; i += stride) {
	        /* Compute forces & integrate for the body */
	        device_bodyForce(coords_ptr, veloc_ptr, dt, bodies_per_system, i);
	        device_integrate(coords_ptr, veloc_ptr, dt, i);
        }
        // sync threads in same block
        __syncthreads();
    }
}

void run_device(coords_T *d_all_coords, velocity_T *d_all_velocities, int num_systems, \
                int bytes_coords, int bytes_velocs, int nIters, int bodies_per_system, \
                float dt) {
    double totalTime;
    cudaError_t runtime_error;
    double interactions_per_system, total_interactions;

    // num_systemsx1 thread blocks, one thread block per galaxy
    dim3 grid_dims(num_systems, 1);
    // 1024x1 threads per block (max for device)
    dim3 block_dims(THREADS_PER_BLOCK, 1);
    
    printf("\n----------------------------\n");
    printf("Running Device simulation for %d systems...\n",
        num_systems);

    totalTime = 0.0;
    StartTimer();

    // Copy data on device
    runtime_error = cudaMemcpy((coords_T *)d_all_coords, (coords_T *)h_data.all_coords, \
                                bytes_coords, cudaMemcpyHostToDevice);
    CHECK_CUDA_FAIL(runtime_error);
    runtime_error = cudaMemcpy((velocity_T *)d_all_velocities, (velocity_T *)h_data.all_velocities, \
                                bytes_velocs, cudaMemcpyHostToDevice);
    CHECK_CUDA_FAIL(runtime_error);

    // kernel launch
    device_nbody<<<grid_dims, block_dims>>>(d_all_coords, d_all_velocities, \
                                            nIters, bodies_per_system, dt);
    cudaDeviceSynchronize();
    runtime_error = cudaGetLastError();
    CHECK_CUDA_FAIL(runtime_error);
    
    // Copy device output back on host
    runtime_error = cudaMemcpy((coords_T *)h_device_out.all_coords, (coords_T *)d_all_coords, \
                                bytes_coords, cudaMemcpyDeviceToHost);
    CHECK_CUDA_FAIL(runtime_error);
    totalTime = GetTimer() / 1000.0;

    /* Metrics calculation */
    interactions_per_system = (double) bodies_per_system * bodies_per_system;
    total_interactions = interactions_per_system * num_systems * nIters;
    printf("Total Time: %.3f seconds\n", totalTime);
    printf("Average Throughput: %0.3f Billion Interactions / second\n\n",
           1e-9 * total_interactions / totalTime);
   
    /* Dump final state of System 0, Body 0 and 1 for verification comparison */
    printf("Final position of System 0, Body 0: %.4f, %.4f, %.4f\n",
           h_device_out.all_coords[0].x, h_device_out.all_coords[0].y, h_device_out.all_coords[0].z);
    printf("Final position of System 0, Body 1: %.4f, %.4f, %.4f\n",
           h_device_out.all_coords[1].x, h_device_out.all_coords[1].y, h_device_out.all_coords[1].z);
    printf("----------------------------\n");
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

    h_data.all_coords = (coords_T *) malloc(bytes_coords);
    h_data.all_velocities = (velocity_T *) malloc(bytes_coords);
    omp_out.all_coords = (coords_T *) malloc(bytes_coords);
    omp_out.all_velocities = (velocity_T *) malloc(bytes_coords);
    h_device_out.all_coords = (coords_T *) malloc(bytes_coords);
    h_device_out.all_velocities = (velocity_T *) malloc(bytes_coords);
    if(IS_NULL(h_data.all_coords) || IS_NULL(h_data.all_velocities) || \
        IS_NULL(omp_out.all_coords) || IS_NULL(omp_out.all_velocities) || \
        IS_NULL(h_device_out.all_coords) || IS_NULL(h_device_out.all_velocities)) {
        printf("Host: Memory allocation failed!\n");
        FREE_MEM(h_data.all_coords);
        FREE_MEM(h_data.all_velocities);   
        FREE_MEM(omp_out.all_coords);
        FREE_MEM(omp_out.all_velocities);
        FREE_MEM(h_device_out.all_coords);
        FREE_MEM(h_device_out.all_velocities);  
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
	// /* Random initialization if file missing */
	//     buf = (float *) h_data;
	//     for (int i = 0; i < 6 * total_bodies; i++) {
	//         buf[i] = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
    //     }
        printf("galaxy_data.bin is missing!\n");
    }

    // Save h_data for transfering to device
    memcpy(omp_out.all_coords, h_data.all_coords, bytes_coords);   
    memcpy(omp_out.all_velocities, h_data.all_velocities, bytes_veloc);

    // Run parallel CPU
    // run_omp(num_systems, nIters, bodies_per_system, dt);
    
    // allocate memory on device
    error_coords = cudaMalloc((void **)&d_data.all_coords, bytes_coords);
    error_velocs = cudaMalloc((void **)&d_data.all_velocities, bytes_veloc);
    if(!IS_CUDA_MALLOC_OK(error_coords) || !IS_CUDA_MALLOC_OK(error_velocs)) {
        printf("CUDA: Memory allocation Failed!\n");
        free(h_data.all_coords);
        free(h_data.all_velocities);   
        free(omp_out.all_coords);
        free(omp_out.all_velocities);
        free(h_device_out.all_coords);
        free(h_device_out.all_velocities); 
        IS_CUDA_MALLOC_OK(error_coords) ? cudaFree(d_data.all_coords) : 0;
        IS_CUDA_MALLOC_OK(error_velocs) ? cudaFree(d_data.all_velocities) : 0;
        cudaDeviceReset();
        return 1;
    }
   
    // Run on GPU
    run_device(d_data.all_coords, d_data.all_velocities ,num_systems, bytes_coords,\
                bytes_veloc, nIters, bodies_per_system, dt);

    /* COMPARE RESULTS */
    printf("Comparing CPU and GPU results...\n");
    coords_T first_dev = h_device_out.all_coords[0];
    coords_T second_dev = h_device_out.all_coords[1];
    coords_T first_omp = omp_out.all_coords[0];
    coords_T second_omp = omp_out.all_coords[1];
    
    float diff_0_x = ABS((first_dev.x - first_omp.x)); 
    float diff_0_y = ABS((first_dev.y - first_omp.y));
    float diff_0_z = ABS((first_dev.z - first_omp.z));
    float diff_1_x = ABS((second_dev.x - second_omp.x));
    float diff_1_y = ABS((second_dev.y - second_omp.y));
    float diff_1_z = ABS((second_dev.z - second_omp.z));

    if (diff_0_x >= 0.001 || diff_0_y >= 0.001 || diff_0_z >= 0.001 ||
        diff_1_x >= 0.001 || diff_1_y >= 0.001 || diff_1_z >= 0.001) {
        printf("Results differ!\n");
        printf("Difference of System 0, Body 0: %.4f, %.4f, %.4f\n", 
        diff_0_x, diff_0_y, diff_0_z);
        printf("Difference of System 0, Body 1: %.4f, %.4f, %.4f\n", 
        diff_1_x, diff_1_y, diff_1_z);
    }
    else 
        printf("Correct!\n");
    
    free(h_data.all_coords);
    free(h_data.all_velocities);   
    free(omp_out.all_coords);
    free(omp_out.all_velocities);
    free(h_device_out.all_coords);
    free(h_device_out.all_velocities); 
    cudaFree(d_data.all_coords);
    cudaFree(d_data.all_velocities);    
    
    cudaDeviceReset();
    return 0;
}