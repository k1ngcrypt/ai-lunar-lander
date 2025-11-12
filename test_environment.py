#!/usr/bin/env python3
"""
test_environment.py
Quick environment setup test for AI Lunar Lander

This script verifies that the environment is set up correctly by:
1. Testing environment creation
2. Validating with SB3 checker
3. Testing environment interaction
4. Testing model creation and mini-training

This test was extracted from unified_training.py for better separation of concerns.

Usage:
    python test_environment.py              # Run quick test
    python test_environment.py --verbose    # Verbose output
"""

import sys
import argparse


def test_environment_setup(verbose=False, seed=42):
    """
    Quick test to verify environment setup (~2 minutes).
    
    Args:
        verbose: Print detailed progress information
        seed: Random seed for reproducibility
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("\n" + "="*80)
    print("QUICK ENVIRONMENT TEST")
    print("="*80)
    print("\nThis will verify that your environment is set up correctly.")
    print("Running 5,000 timesteps (~2 minutes)...")
    print("="*80 + "\n")
    
    try:
        # Import dependencies
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.env_checker import check_env
        from lunar_lander_env import LunarLanderEnv
        
        # Test environment creation
        print("[1/4] Testing environment creation...")
        env = LunarLanderEnv(
            observation_mode='compact',
            max_episode_steps=200,
            create_new_sim_on_reset=True  # Avoid Basilisk warnings
        )
        print("  ✓ Environment created successfully")
        
        # Test environment validation
        print("\n[2/4] Validating environment with SB3 checker...")
        check_env(env, warn=True)
        print("  ✓ Environment validation passed")
        
        # Test reset and step
        print("\n[3/4] Testing environment interaction...")
        obs, info = env.reset(seed=seed)
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        print("  ✓ Environment interaction successful")
        env.close()
        
        # Test model creation and training
        print("\n[4/4] Testing model creation and mini-training...")
        
        def make_env():
            return LunarLanderEnv(
                observation_mode='compact',
                max_episode_steps=200,
                create_new_sim_on_reset=True
            )
        
        env = DummyVecEnv([make_env])
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed
        )
        model.learn(total_timesteps=5_000, progress_bar=not verbose)
        print("  ✓ Model training successful")
        env.close()
        
        # Success!
        print("\n" + "="*80)
        print("✓ ENVIRONMENT TEST PASSED!")
        print("="*80)
        print("\nYour setup is working correctly! You can now:")
        print("  1. Run unit tests: python run_tests.py")
        print("  2. Run demo: python unified_training.py --mode demo")
        print("  3. Start standard training: python unified_training.py --mode standard")
        print("  4. Run full curriculum: python unified_training.py --mode curriculum")
        print("="*80 + "\n")
        
        return True
        
    except Exception as e:
        print("\n" + "="*80)
        print("✗ ENVIRONMENT TEST FAILED")
        print("="*80)
        print(f"\nError: {e}")
        print("\nPlease check:")
        print("  1. Stable Baselines3: pip install stable-baselines3[extra]")
        print("  2. Gymnasium: pip install gymnasium")
        print("  3. NumPy: pip install numpy")
        print("  4. Basilisk simulation is working")
        print("="*80 + "\n")
        
        if verbose:
            import traceback
            traceback.print_exc()
        
        return False


def main():
    """Main entry point for environment test script"""
    parser = argparse.ArgumentParser(
        description='Test AI Lunar Lander environment setup'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output with detailed error messages'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Run test
    success = test_environment_setup(verbose=args.verbose, seed=args.seed)
    
    # Return appropriate exit code
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
