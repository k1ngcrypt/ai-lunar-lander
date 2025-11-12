"""
moon_lander_vizard_b23.py
Real-time lunar landing visualization with dynamic terrain
for Basilisk 2.3 and Vizard
"""

import sys
import numpy as np
import time

# -------------------------
# Basilisk 2.3 Vizard interface
# -------------------------
try:
    from Basilisk import VizInterface
except ImportError:
    print("⚠ Basilisk VizInterface not found. Make sure Basilisk 2.3 is installed.")
    sys.exit(1)

# -------------------------
# Terrain generator
# -------------------------
try:
    from generate_terrain import generate_lunar_terrain
except ImportError:
    print("⚠ generate_terrain.py not found.")
    sys.exit(1)

# -------------------------
# Paths and files
# -------------------------
TERRAIN_SIZE = 2000.0  # meters
RESOLUTION = 200
starship_model = "starship_hls.glb"

# -------------------------
# Always generate new terrain
# -------------------------
print("Generating new lunar terrain...")
heightmap = generate_lunar_terrain(
    size=TERRAIN_SIZE,
    resolution=RESOLUTION,
    num_craters=20,
    terrain_type='mare',
    realism_level='high',
    seed=None  # random terrain each run
)

# -------------------------
# Initialize Vizard (Basilisk 2.3)
# -------------------------
print("Initializing Basilisk Vizard...")
vizard = VizInterface.VizInterface()
vizard.UseVizard = True  # 2.3 uses property style

# Add low-res Moon sphere
moon_radius = 1737.1  # km
moon_scale = 0.001
vizard.addSphere(
    "Moon",
    radius=moon_radius*moon_scale,
    color=[200, 200, 200],
    pos=[0, 0, -50]
)

# -------------------------
# Create terrain mesh in Vizard
# -------------------------
print("Creating terrain mesh...")
x = np.linspace(-TERRAIN_SIZE / 2, TERRAIN_SIZE / 2, RESOLUTION)
y = np.linspace(-TERRAIN_SIZE / 2, TERRAIN_SIZE / 2, RESOLUTION)
X, Y = np.meshgrid(x, y)
Z = heightmap

# Flatten vertices
vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

# Build faces for triangles
faces = []
for i in range(RESOLUTION - 1):
    for j in range(RESOLUTION - 1):
        idx0 = i*RESOLUTION + j
        idx1 = idx0 + 1
        idx2 = idx0 + RESOLUTION
        idx3 = idx2 + 1
        faces.append([idx0, idx1, idx3])
        faces.append([idx0, idx3, idx2])

vizard.addMesh(
    "Terrain",
    vertices=vertices.tolist(),
    faces=faces.tolist(),
    color=[100, 100, 100]
)

# -------------------------
# Load starship model
# -------------------------
if not starship_model:
    print("⚠ Starship model not specified.")
    sys.exit(1)

print(f"Loading starship model {starship_model}...")
vizard.addModel(
    "Starship",
    starship_model,
    scale=1.0,
    pos=[0, 0, 20]
)

# -------------------------
# Lighting
# -------------------------
vizard.addLight(
    "Sun",
    type="directional",
    direction=[1, 1, -1],
    color=[255, 255, 255]
)

# -------------------------
# Real-time update loop
# -------------------------
def simulate_basilisk_positions():
    """
    Simulate incoming Basilisk position data
    """
    t = 0
    while True:
        x_pos = np.sin(t/10) * 100
        y_pos = np.cos(t/10) * 100
        z_pos = 20 + 5*np.sin(t/5)
        t += 0.1
        yield [x_pos, y_pos, z_pos]
        time.sleep(0.05)

positions = simulate_basilisk_positions()
starship_name = "Starship"

print("Starting real-time Vizard visualization loop...")
for pos in positions:
    # Update starship position
    vizard.setPosition(starship_name, pos)
    
    # Camera follows starship
    cam_pos = [pos[0]+50, pos[1]+50, pos[2]+50]
    vizard.setCameraPosition(cam_pos, pos)
    
    vizard.update()
