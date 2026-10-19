#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../timer.h"

#define SOFTENING 0.01f 

typedef struct {
    float x, y, z, vx, vy, vz;
} Body;

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

int main(const int argc, const char *argv[]) {
    /* Default Configuration */
    int num_systems = 32;       	/* Number of independent galaxies */
    int bodies_per_system = 8192;	/* Number of bodies per galaxy */
    int nIters = 20;            	/* Simulation steps */ 
    
    const float dt = 0.01f;
    FILE *fp;
    int total_bodies, bytes, sys, iter;
    Body *data, *system_ptr;
    float *buf;
    double totalTime, interactions_per_system, total_interactions;


    /* Attempt to load dataset */
    fp = fopen("../galaxy_data.bin", "rb");
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
    data = (Body *) malloc(bytes);

    /* Initialize data */
    if (fp) {
	    fread(data, sizeof(Body), total_bodies, fp);
	    fclose(fp);
    } else {
	/* Random initialization if file missing */
	    buf = (float *) data;
	    for (int i = 0; i < 6 * total_bodies; i++) {
	        buf[i] = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
        }
    }

    printf("Running sequential CPU simulation for %d systems...\n",
           num_systems);

    totalTime = 0.0;

    StartTimer();

    /* Time-steps */
    for (iter = 1; iter <= nIters; iter++) {
        /* Galaxies */
	    for (sys = 0; sys < num_systems; sys++) {
	        /* Calculate offset for the galaxy */
	        system_ptr = &data[sys * bodies_per_system];
	        
	        /* Compute forces & integrate for the galaxy */
	        bodyForce(system_ptr, dt, bodies_per_system);
	        integrate(system_ptr, dt, bodies_per_system);
        }
    }

    totalTime = GetTimer() / 1000.0;

    /* Metrics calculation */
    interactions_per_system = (double) bodies_per_system * bodies_per_system;
    total_interactions = interactions_per_system * num_systems * nIters;

    printf("Total Time: %.3f seconds\n", totalTime);
    printf("Average Throughput: %0.3f Billion Interactions / second\n",
           1e-9 * total_interactions / totalTime);

    /* Dump final state of System 0, Body 0 and 1 for verification comparison */
    printf("Final position of System 0, Body 0: %.4f, %.4f, %.4f\n",
           data[0].x, data[0].y, data[0].z);
    printf("Final position of System 0, Body 1: %.4f, %.4f, %.4f\n",
           data[1].x, data[1].y, data[1].z);

    free(data);
    return 0;
}