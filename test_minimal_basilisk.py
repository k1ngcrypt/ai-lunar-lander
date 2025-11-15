"""
Minimal Basilisk test to isolate divergence
Tests spacecraft dynamics WITHOUT custom controllers
"""
import sys
sys.path.insert(0, 'basilisk/dist3')

from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody
from Basilisk.simulation import spacecraft
import numpy as np

# Create minimal simulation
scSim = SimulationBaseClass.SimBaseClass()

# Create process
dynProcess = scSim.CreateNewProcess("dynamics")
dt = 0.01  # seconds
dynProcess.addTask(scSim.CreateNewTask("dynTask", macros.sec2nano(dt)))

# Create spacecraft with minimal mass
lander = spacecraft.Spacecraft()
lander.ModelTag = "TestLander"
lander.hub.mHub = 10000.0  # 10 tons (much lighter than 1.3M kg)
lander.hub.r_BcB_B = np.zeros(3)
lander.hub.IHubPntBc_B = np.diag([10000, 10000, 5000])  # kg·m²

# Initial conditions
lander.hub.r_CN_NInit = np.array([0., 0., 100.0])  # 100m altitude
lander.hub.v_CN_NInit = np.array([0., 0., -5.0])   # -5 m/s descent
lander.hub.sigma_BNInit = np.zeros(3)
lander.hub.omega_BN_BInit = np.zeros(3)

scSim.AddModelToTask("dynTask", lander)

# Add lunar gravity
gravFactory = simIncludeGravBody.gravBodyFactory()
moon = gravFactory.createMoon()
moon.isCentralBody = True
scSim.AddModelToTask("dynTask", gravFactory.spiceObject)

# Connect gravity to spacecraft
lander.gravField.gravBodies = spacecraft.GravBodyVector(list(gravFactory.gravBodies.values()))

# Initialize
scSim.InitializeSimulation()

print("\\n=== MINIMAL BASILISK TEST ===")
print(f"Initial: alt={lander.hub.r_CN_NInit[2]:.2f}m, vel_z={lander.hub.v_CN_NInit[2]:.2f} m/s")
print(f"Mass: {lander.hub.mHub:.0f} kg")

# Step once
scSim.ConfigureStopTime(macros.sec2nano(dt))
scSim.ExecuteSimulation()

# Read state
pos = lander.scStateOutMsg.read().r_BN_N
vel = lander.scStateOutMsg.read().v_BN_N

print(f"After 1 step ({dt}s):")
print(f"  alt={pos[2]:.2f}m, vel_z={vel[2]:.2f} m/s")

# Check for divergence
if abs(pos[2] - 100) < 10:
    print("\\n✓ SUCCESS: No divergence!")
else:
    print(f"\\n✗ FAIL: Divergence detected! (expected ~100m, got {pos[2]:.2f}m)")
