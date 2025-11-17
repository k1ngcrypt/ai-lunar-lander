import sys
import numpy as np

# -------------------------
# Basilisk Vizard interface
# -------------------------
from common_utils import setup_basilisk_path
setup_basilisk_path()

from Basilisk.utilities import SimulationBaseClass, macros
from Basilisk.simulation import spacecraft
from Basilisk.utilities import orbitalMotion

# -------------------------
# Import VizardManager
# -------------------------
from vizard_integration import VizardManager


# -------------------------
# Create Basilisk simulation
# -------------------------

sim = SimulationBaseClass.SimBaseClass()
process = sim.CreateNewProcess("process")
task = sim.CreateNewTask("task", macros.sec2nano(0.1))
process.addTask(task)

# -------------------------
# Initialize Vizard Manager
# -------------------------

viz = VizardManager(port=5570, enabled=True, name="LunarLander")

# Add Vizard interface to the sim
viz.add_to_simulation(sim, "task")

print("🚀 Basilisk Vizard configured on port 5570")
print("👉 Now open Vizard and connect to:  localhost:5570\n")


# -------------------------
# Dummy spacecraft state (same as before)
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
# Run simulation
# -------------------------

print("Initializing Basilisk simulation...")
sim.InitializeSimulation()

print("Running Basilisk for 10 minutes of *simulation time*...")  
print("This keeps the Vizard socket open so you can connect.\n")

# 10 minutes of sim time
TEN_MINUTES = macros.min2nano(10)

sim.ConfigureStopTime(TEN_MINUTES)
sim.ExecuteSimulation()     # NOTE: no arguments ever allowed

print("Simulation finished or time elapsed.")




print("Simulation running! Vizard can now connect.")

# Keep script alive so Vizard doesn't time out
print("Press CTRL+C to quit.")
while True:
    pass

print("✅ Simulation complete. Check Vizard.\n")

# Cleanup explicitly (optional)
viz.close()


