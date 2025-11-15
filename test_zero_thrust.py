"""Test with ZERO thrust to see if thrusters cause divergence"""
import sys
sys.path.insert(0, 'basilisk/dist3')

from lunar_lander_env import LunarLanderEnv
import numpy as np

env = LunarLanderEnv(
    max_episode_steps=10, 
    verbose=0, 
    initial_altitude_range=(70, 80),
    initial_velocity_range=((-5, 5), (-5, 5), (-5, -2)),
    terrain_config={'size': 2000, 'resolution': 200, 'num_craters': 0, 
                    'crater_depth_range': (3, 12), 'crater_radius_range': (15, 60)},
    create_new_sim_on_reset=True
)
obs, info = env.reset()

print(f"\n=== TEST WITH ZERO THRUST ===")
print(f"Initial: alt={obs[2]:.2f}m, vel_z={obs[5]:.2f} m/s")

# Apply ZERO thrust (all zeros)
action_zero = np.zeros(15)
obs, r, t, tr, info = env.step(action_zero)
print(f"Step 1:  alt={obs[2]:.2f}m, vel_z={obs[5]:.2f} m/s")

# Expected: gentle descent under gravity alone
# Moon gravity = 1.62 m/s², so after 0.01s: vel_z should be ~-2.57 - 0.0162 = -2.586 m/s
expected_vel = -2.57 - (1.62 * 0.01)
expected_alt = 77.65 - (2.57 * 0.01)

print(f"\nExpected (gravity only):")
print(f"  vel_z ≈ {expected_vel:.3f} m/s")
print(f"  alt ≈ {expected_alt:.2f} m")

if abs(obs[2] - expected_alt) < 5 and abs(obs[5] - expected_vel) < 1:
    print("\n✓ SUCCESS: Physics behaves correctly with zero thrust!")
else:
    print(f"\n✗ FAIL: Still diverging even with zero thrust!")
    print(f"  This proves thrusters are NOT the issue - something else is wrong")
