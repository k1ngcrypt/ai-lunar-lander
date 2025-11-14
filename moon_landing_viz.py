import sys
import numpy as np

# -------------------------
# Basilisk Vizard interface
# -------------------------
from common_utils import setup_basilisk_path
setup_basilisk_path()

from Basilisk.utilities import SimulationBaseClass, macros
from Basilisk.simulation import vizInterface

# -------------------------
# Create Basilisk simulation
# -------------------------

sim = SimulationBaseClass.SimBaseClass()
process = sim.CreateNewProcess("process")
task = sim.CreateNewTask("task", macros.sec2nano(0.1))
process.addTask(task)

# -------------------------
# Add Vizard interface
# -------------------------

vizard = vizInterface.VizInterface()
vizard.pubPortNumber = "5556"

sim.AddModelToTask("task", vizard)

print("🚀 Basilisk Vizard configured on port 5556")
print("👉 Now open Vizard and connect to:  localhost:5556")

# -------------------------
# Dummy spacecraft state (for demo)
# -------------------------

from Basilisk.simulation import spacecraft
from Basilisk.utilities import orbitalMotion

sc = spacecraft.Spacecraft()
sim.AddModelToTask("task", sc)

# orbital state example
from Basilisk.utilities import orbitalMotion

mu = 4.9048695e12

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

sim.InitializeSimulation()
sim.ConfigureStopTime(macros.sec2nano(30.0))

print("▶ Running Basilisk simulation...")
sim.ExecuteSimulation()

print("✅ Simulation complete. Check Vizard.")