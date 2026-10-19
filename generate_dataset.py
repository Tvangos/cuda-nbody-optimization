import struct
import random
import math

# Configuration
NUM_SYSTEMS = 32        # Number of independent galaxies
BODIES_PER_SYSTEM = 8192 # Number of bodies per galaxy
FILENAME = "galaxy_data.bin"

def generate_stable_system(offset_base):
    """
    Generates a system where bodies orbit a heavy center (like a galaxy).
    This allows visual or mathematical verification.
    """
    bodies = []
    
    # 1. Massive black hole in the center
    center_mass = 1000.0
    bodies.append({
        'x': 0.0, 'y': 0.0, 'z': 0.0,
        'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
        'mass': center_mass
    })
    
    # 2. Orbiting stars
    for _ in range(BODIES_PER_SYSTEM - 1):
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(50.0, 500.0)
        velocity = math.sqrt(center_mass / dist) # v = sqrt(GM/r), simplified G=1
        
        # Position
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        z = random.uniform(-10, 10) # Disc-like
        
        # Velocity (Tangential)
        vx = -velocity * math.sin(angle)
        vy = velocity * math.cos(angle)
        vz = 0.0
        
        bodies.append({
            'x': x, 'y': y, 'z': z,
            'vx': vx, 'vy': vy, 'vz': vz,
            'mass': 1.0
        })

    return bodies

with open(FILENAME, 'wb') as f:
    # Write Header: Num Systems, Bodies per System
    f.write(struct.pack('ii', NUM_SYSTEMS, BODIES_PER_SYSTEM))
    
    for i in range(NUM_SYSTEMS):
        system_data = generate_stable_system(i)
        for b in system_data:
            # Write x, y, z, vx, vy, vz (floats)
            # Corresponds to the C struct alignment
            data = struct.pack('ffffff', 
                               b['x'], b['y'], b['z'], 
                               b['vx'], b['vy'], b['vz'])
            f.write(data)

print(f"Generated {FILENAME} with {NUM_SYSTEMS} systems of {BODIES_PER_SYSTEM} bodies.")
