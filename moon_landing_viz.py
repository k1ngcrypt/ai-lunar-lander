"""
moon_landing_viz.py
Real-time lunar landing visualization with dynamic terrain
for Basilisk + Vizard
Adapted to use VizardSettings from Basilisk documentation.
"""

import sys
import numpy as np
import time

# -------------------------
# Basilisk Vizard interface
# -------------------------
from common_utils import setup_basilisk_path
setup_basilisk_path()

from Basilisk.utilities import SimulationBaseClass, macros
from Basilisk.simulation import spacecraft
from Basilisk.utilities import orbitalMotion, vizSupport  # use vizSupport helper as documented

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
    seed=None
)

# -------------------------
# Create Basilisk simulation
# -------------------------
sim = SimulationBaseClass.SimBaseClass()
process = sim.CreateNewProcess("process")
task = sim.CreateNewTask("task", macros.sec2nano(0.1))
process.addTask(task)

# -------------------------
# Dummy spacecraft state
# -------------------------
sc = spacecraft.Spacecraft()
sim.AddModelToTask("task", sc)

mu = 4.9048695e12  # Moon GM
oe = orbitalMotion.ClassicElements()
oe.a = 20000.0 * 1000
oe.e = 0.1
oe.i = 33.3 * macros.D2R
oe.Omega = 48.2 * macros.D2R
oe.omega = 347.8 * macros.D2R
oe.f = 103.5 * macros.D2R

r, v = orbitalMotion.elem2rv(mu, oe)
sc.hub.r_CN_NInit = r
sc.hub.v_CN_NInit = v

# -------------------------
# Initialize Vizard visualization
# -------------------------
# Use vizSupport helper function to enable Unity visualization with settings
print("Initializing Basilisk + Vizard interface...")

viz = vizSupport.enableUnityVisualization(
    scSim     = sim,
    simTaskName = "task",
    scList    = sc,
    liveStream   = True,
    broadcastStream = True,
    noDisplay = False
)

# Apply custom settings as per VizardSettings documentation
viz.settings.ambient = 0.6
viz.settings.orbitLinesOn = 1
viz.settings.spacecraftCSon = 1
viz.settings.planetCSon = 1
viz.settings.customGUIScale = -1
viz.settings.skyBox = "black"

# Optionally set spacecraft sprite (if model not loaded)
viz.settings.defaultSpacecraftSprite = "bskSat"
viz.settings.showSpacecraftAsSprites = -1

# Load starship model if available
if starship_model:
    print(f"Loading starship model: {starship_model}")
    vizSupport.createCustomModel(
        viz,
        modelPath = starship_model,
        scale     = [2.0, 2.0, 2.0]
    )

# -------------------------
# Run simulation to keep Vizard connected
# -------------------------
print("Running simulation so Vizard can connect...")
# Configure a long stop time so visualization socket stays open
sim.ConfigureStopTime(macros.min2nano(10))
sim.ExecuteSimulation()

print("✅ Simulation complete (or 10 min sim time configured).")
print("Now open Vizard and connect to the streaming port shown.")
