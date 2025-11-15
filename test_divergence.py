"""Quick test to diagnose simulation divergence"""
import sys
sys.path.insert(0, 'basilisk/dist3')

from lunar_lander_env import LunarLanderEnv
import numpy as np

# Test with GENTLE initial conditions (low velocity) and NO TERRAIN
env = LunarLanderEnv(
    max_episode_steps=10, 
    verbose=0, 
    initial_altitude_range=(70, 80),
    initial_velocity_range=((-5, 5), (-5, 5), (-5, -2)),  # MUCH slower velocities
    terrain_config={'size': 2000, 'resolution': 200, 'num_craters': 0, 
                    'crater_depth_range': (3, 12), 'crater_radius_range': (15, 60)},
    create_new_sim_on_reset=True
)
obs, info = env.reset()

print(f"\n=== TEST RESULTS (dt=0.01s, NO TERRAIN INTERACTION) ===")
print(f"Initial: alt={obs[2]:.2f}m, vel_z={obs[5]:.2f} m/s")

# Test with hovering thrust (adjusted for new throttle limits)
action_hover = np.array([0.3]*3 + [0]*12)  # 30% throttle (within 15-60% range)
obs, r, t, tr, info = env.step(action_hover)
print(f"Step 1:  alt={obs[2]:.2f}m, vel_z={obs[5]:.2f} m/s")

if abs(obs[2] - 70) < 20:  # Within 20m of expected
    print("\n✓ SUCCESS: No significant divergence!")
else:
    print(f"\n✗ FAIL: Divergence detected (expected ~70m, got {obs[2]:.2f}m)")


