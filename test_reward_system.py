"""
Quick test script to verify reward system and initial conditions
This helps diagnose why rewards are stuck at -807
"""

import numpy as np
from lunar_lander_env import LunarLanderEnv

def test_basic_spawn():
    """Test that agent doesn't crash immediately on spawn"""
    print("\n" + "="*60)
    print("TEST 1: Basic Spawn (No Actions)")
    print("="*60)
    
    env = LunarLanderEnv(
        observation_mode='compact',
        max_episode_steps=100,
        initial_altitude_range=(50.0, 100.0),
        initial_velocity_range=((-5.0, 5.0), (-5.0, 5.0), (-5.0, -2.0)),
        verbose=1  # Enable debug output
    )
    
    obs, info = env.reset(seed=42)
    print(f"\nInitial state:")
    print(f"  Altitude: {info['initial_altitude']:.2f}m")
    print(f"  Position: {info['initial_position']}")
    print(f"  Velocity: {info['initial_velocity']}")
    
    # Take 5 random actions
    total_reward = 0
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"\nStep {i+1}:")
        print(f"  Altitude: {info['altitude']:.2f}m")
        print(f"  Reward: {reward:.2f} (total: {total_reward:.2f})")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        
        if terminated or truncated:
            print(f"\n  Episode ended!")
            print(f"  Final reward components: {info['reward_components']}")
            break
    
    env.close()
    return total_reward

def test_hovering():
    """Test that hovering gives reasonable rewards"""
    print("\n" + "="*60)
    print("TEST 2: Hovering at 50m")
    print("="*60)
    
    env = LunarLanderEnv(
        observation_mode='compact',
        max_episode_steps=50,
        initial_altitude_range=(50.0, 55.0),  # Start at ~50m
        initial_velocity_range=((-1.0, 1.0), (-1.0, 1.0), (-2.0, 0.0)),
        verbose=0  # Disable per-step debug
    )
    
    obs, info = env.reset(seed=42)
    
    # Hover action: moderate throttle, no torques
    hover_action = np.array([
        0.7, 0.7, 0.7,  # Primary throttles (70%)
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Gimbals (neutral)
        0.0, 0.0, 0.0,  # Mid-body (off)
        0.0, 0.0, 0.0   # RCS (off)
    ])
    
    total_reward = 0
    for i in range(20):
        obs, reward, terminated, truncated, info = env.step(hover_action)
        total_reward += reward
        
        if i % 5 == 0:
            print(f"Step {i}: altitude={info['altitude']:.2f}m, reward={reward:.2f}, total={total_reward:.2f}")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {i}")
            print(f"Reward components: {info['reward_components']}")
            break
    
    env.close()
    return total_reward

def test_reward_components():
    """Test reward component breakdown"""
    print("\n" + "="*60)
    print("TEST 3: Reward Component Analysis")
    print("="*60)
    
    env = LunarLanderEnv(
        observation_mode='compact',
        max_episode_steps=20,
        initial_altitude_range=(30.0, 35.0),
        initial_velocity_range=((-2.0, 2.0), (-2.0, 2.0), (-3.0, -1.0)),
        verbose=1  # Enable per-step debug
    )
    
    obs, info = env.reset(seed=42)
    
    # Gentle descent action
    descent_action = np.array([
        0.5, 0.5, 0.5,  # Moderate throttle
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ])
    
    total_reward = 0
    for i in range(10):
        obs, reward, terminated, truncated, info = env.step(descent_action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"\nFinal reward components:")
            for comp, value in sorted(info['reward_components'].items(), 
                                     key=lambda x: abs(x[1]), reverse=True):
                print(f"  {comp}: {value:.2f}")
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    env.close()
    return total_reward

if __name__ == "__main__":
    print("\n" + "="*80)
    print("REWARD SYSTEM DIAGNOSTIC TEST")
    print("="*80)
    
    try:
        r1 = test_basic_spawn()
        print(f"\nTest 1 result: {r1:.2f}")
    except Exception as e:
        print(f"\nTest 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        r2 = test_hovering()
        print(f"\nTest 2 result: {r2:.2f}")
    except Exception as e:
        print(f"\nTest 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        r3 = test_reward_components()
        print(f"\nTest 3 result: {r3:.2f}")
    except Exception as e:
        print(f"\nTest 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
