from lunar_lander_env import LunarLanderEnv
import numpy as np

env = LunarLanderEnv(
    observation_mode='compact',
    max_episode_steps=10,
    initial_altitude_range=(50.0, 60.0),
    initial_velocity_range=((-2.0, 2.0), (-2.0, 2.0), (-3.0, -1.0))
)

obs, info = env.reset(seed=42)
print(f"Initial altitude: {info['initial_altitude']:.2f}m")
print(f"Initial position: [{info['initial_position'][0]:.2f}, {info['initial_position'][1]:.2f}, {info['initial_position'][2]:.2f}]")

for i in range(3):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}: altitude={info['altitude']:.2f}m, reward={reward:.2f}, term={terminated}")
    if terminated or truncated:
        break

env.close()
print("Test complete - no divergence!")
