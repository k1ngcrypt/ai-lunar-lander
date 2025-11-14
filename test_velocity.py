"""
Quick test script to verify velocity initialization and reset behavior
"""
import numpy as np
from lunar_lander_env import LunarLanderEnv

print("="*60)
print("VELOCITY INITIALIZATION TEST")
print("="*60)

# Create environment
env = LunarLanderEnv()

# First reset
print("\n[TEST 1] First reset:")
obs, info = env.reset(seed=42)
print(f"  Position: {info['initial_position']}")
print(f"  Velocity: {info['initial_velocity']}")
print(f"  Velocity magnitude: {np.linalg.norm(info['initial_velocity']):.2f} m/s")

# Take one step
print("\n[TEST 2] After 1 step:")
obs, reward, term, trunc, info = env.step(env.action_space.sample())
print(f"  Velocity: {info['velocity']}")
print(f"  Velocity magnitude: {np.linalg.norm(info['velocity']):.2f} m/s")

# Second reset (should have different random velocity)
print("\n[TEST 3] Second reset:")
obs, info = env.reset()
print(f"  Position: {info['initial_position']}")
print(f"  Velocity: {info['initial_velocity']}")
print(f"  Velocity magnitude: {np.linalg.norm(info['initial_velocity']):.2f} m/s")

# Third reset with same seed (should match first reset)
print("\n[TEST 4] Third reset with same seed (should match TEST 1):")
obs, info = env.reset(seed=42)
print(f"  Position: {info['initial_position']}")
print(f"  Velocity: {info['initial_velocity']}")
print(f"  Velocity magnitude: {np.linalg.norm(info['initial_velocity']):.2f} m/s")

# Test multiple resets to check for velocity corruption
print("\n[TEST 5] Multiple resets to check for state corruption:")
for i in range(5):
    obs, info = env.reset()
    vel_mag = np.linalg.norm(info['initial_velocity'])
    print(f"  Reset {i+1}: velocity magnitude = {vel_mag:.2f} m/s")
    # Check if velocity is within expected range
    if vel_mag > 250.0:  # Expected max is ~223 m/s (sqrt(200^2 + 200^2 + 100^2))
        print(f"    ⚠ WARNING: Velocity magnitude ({vel_mag:.2f}) exceeds expected range!")
    else:
        print(f"    ✓ Velocity within expected range")

env.close()

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
