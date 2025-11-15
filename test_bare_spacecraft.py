"""
Test if fuel tanks cause divergence
Minimal spacecraft: just hub + gravity (no tanks, no thrusters)
"""
import sys
sys.path.insert(0, 'basilisk/dist3')

from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody
from Basilisk.simulation import spacecraft
import starship_constants as SC
import numpy as np

print("\n" + "="*70)
print("MINIMAL TEST: Spacecraft with ONLY hub + gravity")
print("="*70)

scSim = SimulationBaseClass.SimBaseClass()
dynProcess = scSim.CreateNewProcess("dynamics")
dt = 0.01
dynProcess.addTask(scSim.CreateNewTask("dynTask", macros.sec2nano(dt)))

# Create bare spacecraft (no effectors at all)
lander = spacecraft.Spacecraft()
lander.ModelTag = "MinimalLander"
lander.hub.mHub = SC.HUB_MASS  # 10,500 kg
lander.hub.r_BcB_B = SC.CENTER_OF_MASS_OFFSET
lander.hub.IHubPntBc_B = SC.INERTIA_TENSOR_FULL
MOON_RADIUS = 1737400.0  # meters (position is relative to Moon's center!)
lander.hub.r_CN_NInit = np.array([0., 0., MOON_RADIUS + 80.0])
lander.hub.v_CN_NInit = np.array([0., 0., -4.0])
lander.hub.sigma_BNInit = np.zeros(3)
lander.hub.omega_BN_BInit = np.zeros(3)

scSim.AddModelToTask("dynTask", lander)

# Add lunar gravity
gravFactory = simIncludeGravBody.gravBodyFactory()
moon = gravFactory.createMoon()
moon.isCentralBody = True
gravFactory.addBodiesTo(lander)

# Initialize
scSim.InitializeSimulation()

print(f"Initial: alt=80.0m above surface (pos_z={MOON_RADIUS + 80.0:.0f}m from center), vel_z=-4.0 m/s")
print(f"Mass: {lander.hub.mHub:.0f} kg")
inertia_val = SC.INERTIA_TENSOR_FULL[0,0]
print(f"Inertia: {inertia_val:.1e} kg·m²")
print("Effectors: NONE (bare spacecraft)")

# Step simulation
scSim.ConfigureStopTime(macros.sec2nano(dt))
scSim.ExecuteSimulation()

# Read state
pos = lander.scStateOutMsg.read().r_BN_N
vel = lander.scStateOutMsg.read().v_BN_N

print(f"\nAfter {dt}s step:")
alt_above_surface = pos[2] - MOON_RADIUS
print(f"  alt_above_surface={alt_above_surface:.2f}m, pos_z={pos[2]:.0f}m, vel_z={vel[2]:.2f} m/s")

# Expected with gravity only: vel should change by -1.62 * 0.01 = -0.0162 m/s
expected_vel = -4.0 - (1.62 * 0.01)
expected_alt = (MOON_RADIUS + 80.0) - (4.0 * 0.01)  # Position includes Moon radius!

print(f"\nExpected (gravity only):")
print(f"  vel_z ≈ {expected_vel:.4f} m/s")
expected_alt_above_surface = expected_alt - MOON_RADIUS
print(f"  alt_above_surface ≈ {expected_alt_above_surface:.4f} m (pos_z ≈ {expected_alt:.0f}m)")

if abs(alt_above_surface - expected_alt_above_surface) < 1.0 and abs(vel[2] - expected_vel) < 0.1:
    print("\n✓ SUCCESS: Physics behaves correctly!")
    print("   Root cause was: Position is relative to Moon's CENTER, not surface!")
else:
    print(f"\n✗ FAIL: Still diverging!")
    print(f"   Altitude error: {abs(alt_above_surface - expected_alt_above_surface):.2f}m")
    print(f"   Velocity error: {abs(vel[2] - expected_vel):.4f} m/s")
