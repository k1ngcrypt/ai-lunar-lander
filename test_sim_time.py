"""Test to verify simulation timestep is correct"""
import sys
sys.path.insert(0, 'basilisk/dist3')

from lunar_lander_env import LunarLanderEnv
import numpy as np

env = LunarLanderEnv(max_episode_steps=5, verbose=0)
obs, info = env.reset()

print(f"\nChecking simulation time progression:")
print(f"  dt = {env.dt} seconds")
print(f"  current_step after reset: {env.current_step}")

# Get initial time
from Basilisk.utilities import macros
initial_time = env.scSim.TotalSim.CurrentNanos / macros.NANO2SEC
print(f"  Basilisk sim time after reset: {initial_time:.6f}s")

# Take one step
action = np.array([0.3]*3 + [0]*12)
print(f"  About to step with sim_time={env.sim_time:.6f}s")
obs, r, t, tr, info = env.step(action)
print(f"  After step, sim_time={env.sim_time:.6f}s")

# Check time after step
after_time = env.scSim.TotalSim.CurrentNanos / macros.NANO2SEC
time_delta = after_time - initial_time

print(f"  current_step after 1 step: {env.current_step}")
print(f"  Basilisk sim time after step: {after_time:.6f}s")
print(f"  Time advanced: {time_delta:.6f}s (expected {env.dt:.6f}s)")

if abs(time_delta - env.dt) < 0.001:
    print("\\n✓ Timestep is correct!")
else:
    print(f"\\n✗ ERROR: Timestep mismatch! Simulation ran for {time_delta:.6f}s instead of {env.dt:.6f}s")
