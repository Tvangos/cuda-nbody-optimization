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
#define CHECK_CUDA_FAIL(err)\  
({\
  if (err != cudaSuccess) {\
    printf("CUDA error: %s (%d) in %s, line %i\n", cudaGetErrorString(err), err, __FILE__, __LINE__);\
    /* free host mem */\
    FREE_MEM(h_data);\
    FREE_MEM(h_device_out);\
    FREE_MEM(omp_out);\
    /* free device mem */\
    cudaFree(d_data);\
    /* Reset device just in case */\
    cudaDeviceReset();\
    exit(1);\
  } \
})

typedef struct {
    float x, y, z, vx, vy, vz;
} Body;

Body *h_data, *h_device_out, *omp_out, *d_data;

/* Update a single galaxy. Parameters:
    - array of bodies
    - time step
    - number of bodies
*/
void bodyForce(Body * p, float dt, int n) {
    int i, j;
    float Fx, Fy, Fz, dx, dy, dz, distSqr, invDist, invDist3;

    for (i = 0; i < n; i++) {
	    Fx = 0.0f;
    	Fy = 0.0f;
    	Fz = 0.0f;

    	for (j = 0; j < n; j++) {
	        dx = p[j].x - p[i].x;
            dy = p[j].y - p[i].y;
            dz = p[j].z - p[i].z;
            distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            invDist = 1.0f / sqrtf(distSqr);
            invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}

/* Integrate positions.
    - array of bodies
    - time step
    - number of bodies
*/
void integrate(Body * p, float dt, int n) {
    int i;
    
    for (i = 0; i < n; i++) {
	    p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

/* Update a single body in a galaxy. Parameters:
    - array of bodies
    - time step
    - number of bodies
    - body index in data array
*/
__device__ void device_bodyForce(Body * p, float dt, int n, int bodyIdx) {
    int j;
    float Fx, Fy, Fz, dx, dy, dz, distSqr, invDist, invDist3;

    Fx = 0.0f;
    Fy = 0.0f;
    Fz = 0.0f;

    for (j = 0; j < n; j++) {
	    dx = p[j].x - p[bodyIdx].x;
        dy = p[j].y - p[bodyIdx].y;
        dz = p[j].z - p[bodyIdx].z;
        distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        invDist = 1.0f / sqrtf(distSqr);
        invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    p[bodyIdx].vx += dt * Fx;
    p[bodyIdx].vy += dt * Fy;
    p[bodyIdx].vz += dt * Fz;
}

/* Integrate position of single body.
    - array of bodies
    - time step
    - body index in data array
*/
__device__ void device_integrate(Body * p, float dt, int bodyIdx) {

	p[bodyIdx].x += p[bodyIdx].vx * dt;
    p[bodyIdx].y += p[bodyIdx].vy * dt;
    p[bodyIdx].z += p[bodyIdx].vz * dt;
}

__global__ void device_nbody(Body *d_data, int nIters, int bodies_per_system, float dt) {
    int iter, i;
    int sys = blockIdx.x; 
    int stride = blockDim.x;

    /* Calculate offset for the galaxy */
    Body *system_ptr = &d_data[sys * bodies_per_system];
    
    /* Time-steps */        
    for (iter = 1; iter <= nIters; iter++) {
        // 1 thread for bodies_per_system / THREADS_PER_BLOCK bodies
        for (i = threadIdx.x; i < bodies_per_system; i += stride) {
	        /* Compute forces & integrate for the body */
	        device_bodyForce(system_ptr, dt, bodies_per_system, i);
	        device_integrate(system_ptr, dt, i);
        }
        // sync threads in same block
        __syncthreads();
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
            Body *system_ptr = &omp_out[sys * bodies_per_system];
	        
	        /* Compute forces & integrate for the galaxy */
	        bodyForce(system_ptr, dt, bodies_per_system);
	        integrate(system_ptr, dt, bodies_per_system);
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
           omp_out[0].x, omp_out[0].y, omp_out[0].z);
    printf("Final position of System 0, Body 1: %.4f, %.4f, %.4f\n",
           omp_out[1].x, omp_out[1].y, omp_out[1].z);
    printf("----------------------------\n");
}

/* GPU CODE */
void run_device(int num_systems, int bytes, int nIters, int bodies_per_system, \
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
    runtime_error = cudaMemcpy((Body *)d_data, (Body *)h_data, bytes, \
                                cudaMemcpyHostToDevice);
    CHECK_CUDA_FAIL(runtime_error);

    // kernel launch
    device_nbody<<<grid_dims, block_dims>>>(d_data, nIters, bodies_per_system, dt);
    cudaDeviceSynchronize();
    runtime_error = cudaGetLastError();
    CHECK_CUDA_FAIL(runtime_error);
    
    // Copy device output back on host
    runtime_error = cudaMemcpy((Body *)h_device_out, (Body *)d_data, bytes, \
                                cudaMemcpyDeviceToHost);
    CHECK_CUDA_FAIL(runtime_error);
    
    /* Metrics calculation */
    interactions_per_system = (double) bodies_per_system * bodies_per_system;
    total_interactions = interactions_per_system * num_systems * nIters;
    totalTime = GetTimer() / 1000.0;
    printf("Total Time: %.3f seconds\n", totalTime);
    printf("Average Throughput: %0.3f Billion Interactions / second\n\n",
           1e-9 * total_interactions / totalTime);
   
    /* Dump final state of System 0, Body 0 and 1 for verification comparison */
    printf("Final position of System 0, Body 0: %.4f, %.4f, %.4f\n",
           h_device_out[0].x, h_device_out[0].y, h_device_out[0].z);
    printf("Final position of System 0, Body 1: %.4f, %.4f, %.4f\n",
           h_device_out[1].x, h_device_out[1].y, h_device_out[1].z);
    printf("----------------------------\n");
}

int main(const int argc, const char *argv[]) {
    /* Default Configuration */
    int num_systems = 32;       	/* Number of independent galaxies */
    int bodies_per_system = 8192;	/* Number of bodies per galaxy */
    int nIters = 20;            	/* Simulation steps */ 
    
    const float dt = 0.01f;
    FILE *fp;
    int total_bodies, bytes;
    float *buf;

    cudaError_t error_data;

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
    bytes = total_bodies * sizeof(Body);
    h_data = (Body *) malloc(bytes);
    omp_out = (Body *) malloc(bytes);
    h_device_out = (Body *) malloc(bytes);

    /* Initialize data */
    if (fp) {
	    fread(h_data, sizeof(Body), total_bodies, fp);
	    fclose(fp);
    } else {
	/* Random initialization if file missing */
	    buf = (float *) h_data;
	    for (int i = 0; i < 6 * total_bodies; i++) {
	        buf[i] = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
        }
    }

    // Save h_data for transfering to device
    memcpy(omp_out, h_data, bytes);   
    
    // Run parallel CPU
    // run_omp(num_systems, nIters, bodies_per_system, dt);
    
    // allocate memory on device
    error_data = cudaMalloc((void **)&d_data, bytes);
    if(!IS_CUDA_MALLOC_OK(error_data)) {
        printf("CUDA: Memory allocation Failed!\n");
        FREE_MEM(h_data);
        FREE_MEM(omp_out);
        FREE_MEM(h_device_out);
        cudaFree(d_data);
        cudaDeviceReset();
        return 1;
    }
   
    // Run on GPU
    run_device(num_systems, bytes, nIters, bodies_per_system, dt);

    /* COMPARE RESULTS */
    printf("Comparing CPU and GPU results...\n");
    float diff_0_x = ABS((h_device_out[0].x - omp_out[0].x)); 
    float diff_0_y = ABS((h_device_out[0].y - omp_out[0].y));
    float diff_0_z = ABS((h_device_out[0].z - omp_out[0].z));
    float diff_1_x = ABS((h_device_out[1].x - omp_out[1].x));
    float diff_1_y = ABS((h_device_out[1].y - omp_out[1].y));
    float diff_1_z = ABS((h_device_out[1].z - omp_out[1].z));

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
    
    free(h_data);
    free(omp_out);
    free(h_device_out);
    cudaFree(d_data);
    cudaDeviceReset();

    return 0;
}