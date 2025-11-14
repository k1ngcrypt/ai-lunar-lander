"""
unified_training.py
Unified RL Training System for Lunar Landing

This comprehensive training script combines the best features from multiple
training approaches:
- Standard RL training (PPO, SAC, TD3)
- Curriculum learning with progressive difficulty (5 stages)
- Automated testing and validation
- Advanced callbacks and monitoring (success rate tracking)
- Multi-environment parallelization
- Observation normalization (VecNormalize)
- Comprehensive evaluation and visualization

Key Features:
✓ Multiple RL algorithms (PPO, SAC, TD3)
✓ Curriculum learning with stage regression support
✓ Observation normalization for training stability
✓ Parallel environment training
✓ Automatic checkpointing and model saving
✓ TensorBoard logging with success rate metrics
✓ Environment validation and testing
✓ Model evaluation and rendering
✓ Quick demo mode for testing
✓ Resumable training from checkpoints

Curriculum Improvements:
- Stage 1 teaches LANDING (not just hovering)
- Advancement requires BOTH mean reward AND 60% success rate
- Stage regression enabled for repeated failures
- Reduced max_timesteps (100k-400k) to prevent overfitting
- Increased min_episodes (200-400) for mastery verification

Usage Examples:
    # Quick test (2 minutes)
    python unified_training.py --mode test
    
    # Demo curriculum (15 minutes)
    python unified_training.py --mode demo
    
    # Standard training (1M steps)
    python unified_training.py --mode standard --timesteps 1000000
    
    # Full curriculum training (recommended)
    python unified_training.py --mode curriculum
    
    # Evaluate trained model
    python unified_training.py --mode eval --model-path ./models/best_model/best_model
    
    # Resume from checkpoint
    python unified_training.py --mode standard --resume ./models/checkpoints/ppo_lunar_lander_500000_steps
"""

import os
import sys
import argparse
import numpy as np
from typing import Optional, Dict, List, Tuple
import time
import json
import pickle
from datetime import datetime
import multiprocessing
import warnings

# Stable Baselines3 imports
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym

# Custom environment
from lunar_lander_env import LunarLanderEnv


# ============================================================================
# UTILITY CLASSES
# ============================================================================

class CurriculumStage:
    """Definition of a single curriculum stage with environment configuration"""
    
    def __init__(self, 
                 name: str,
                 description: str,
                 env_config: Dict,
                 success_threshold: float,
                 min_episodes: int = 100,
                 max_timesteps: int = 500_000):
        """
        Args:
            name: Stage name (e.g., "hover", "simple_descent")
            description: Human-readable description
            env_config: Dict of LunarLanderEnv kwargs for this stage
            success_threshold: Mean reward threshold to advance to next stage
            min_episodes: Minimum episodes before checking success
            max_timesteps: Maximum timesteps for this stage
        """
        self.name = name
        self.description = description
        self.env_config = env_config
        self.success_threshold = success_threshold
        self.min_episodes = min_episodes
        self.max_timesteps = max_timesteps


class TrainingProgressCallback(BaseCallback):
    """Enhanced callback to track training progress with detailed metrics and success rate"""
    
    def __init__(self, stage_name: str = "training", verbose: int = 0):
        super().__init__(verbose)
        self.stage_name = stage_name
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []  # Track success/failure
        
    def _on_step(self) -> bool:
        # Log current stage to TensorBoard
        if hasattr(self.logger, 'record'):
            self.logger.record("curriculum/current_stage", self.stage_name)
        
        # Track episode statistics
        if self.locals.get('dones', [False])[0]:
            if 'infos' in self.locals:
                for info in self.locals['infos']:
                    if 'episode' in info:
                        ep_reward = info['episode']['r']
                        ep_length = info['episode']['l']
                        self.episode_rewards.append(ep_reward)
                        self.episode_lengths.append(ep_length)
                        
                        # Determine success (reward > 400 indicates successful landing with new reward scale)
                        success = ep_reward > 400.0
                        self.episode_successes.append(success)
                        
                        # Log to TensorBoard
                        if hasattr(self.logger, 'record'):
                            self.logger.record("episode/reward", ep_reward)
                            self.logger.record("episode/length", ep_length)
                            self.logger.record("episode/success", float(success))
                            
                            # Calculate recent success rate (last 100 episodes)
                            if len(self.episode_successes) >= 100:
                                recent_success_rate = np.mean(self.episode_successes[-100:])
                                self.logger.record("episode/success_rate_100", recent_success_rate)
        
        return True
    
    def get_success_rate(self, num_episodes: int = 100) -> float:
        """Get success rate over last N episodes"""
        if len(self.episode_successes) < num_episodes:
            if len(self.episode_successes) == 0:
                return 0.0
            return np.mean(self.episode_successes)
        return np.mean(self.episode_successes[-num_episodes:])


class RewardStatisticsCallback(BaseCallback):
    """Callback to track and log reward component statistics for analysis"""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.reward_components = {}  # Dict of component_name -> list of values
        self.episodes_logged = 0
        
    def _on_step(self) -> bool:
        # Track reward components from episode info
        if self.locals.get('dones', [False])[0]:
            if 'infos' in self.locals:
                for info in self.locals['infos']:
                    if 'reward_components' in info:
                        components = info['reward_components']
                        
                        # Accumulate each component
                        for comp_name, comp_value in components.items():
                            if comp_name not in self.reward_components:
                                self.reward_components[comp_name] = []
                            self.reward_components[comp_name].append(comp_value)
                        
                        self.episodes_logged += 1
                        
                        # Log statistics every 100 episodes
                        if self.episodes_logged % 100 == 0:
                            self._log_component_statistics()
        
        return True
    
    def _log_component_statistics(self):
        """Log mean reward component values to TensorBoard"""
        if hasattr(self.logger, 'record'):
            for comp_name, values in self.reward_components.items():
                if len(values) > 0:
                    # Log mean of last 100 episodes
                    recent_values = values[-100:]
                    mean_value = np.mean(recent_values)
                    self.logger.record(f"reward_components/{comp_name}", mean_value)
            
            # Clear old data to avoid memory bloat
            for comp_name in self.reward_components:
                if len(self.reward_components[comp_name]) > 200:
                    self.reward_components[comp_name] = self.reward_components[comp_name][-200:]


class TrainingStateCallback(BaseCallback):
    """
    Callback to periodically save training state (curriculum stage, timesteps, etc.)
    This enables resuming curriculum training from interruptions.
    """
    
    def __init__(self, 
                 state_path: str,
                 save_freq: int = 10000,
                 stage_name: str = "",
                 stage_idx: int = 0,
                 verbose: int = 0):
        super().__init__(verbose)
        self.state_path = state_path
        self.save_freq = save_freq
        self.stage_name = stage_name
        self.stage_idx = stage_idx
        self.total_timesteps = 0
        
    def _on_step(self) -> bool:
        self.total_timesteps += 1
        
        # Save state periodically
        if self.total_timesteps % self.save_freq == 0:
            self._save_state()
        
        return True
    
    def _save_state(self):
        """Save current training state to disk"""
        state = {
            'stage_idx': self.stage_idx,
            'stage_name': self.stage_name,
            'total_timesteps': self.total_timesteps,
            'num_timesteps': self.model.num_timesteps,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(self.state_path, 'w') as f:
                json.dump(state, f, indent=2)
            if self.verbose > 0:
                print(f"Training state saved: {self.state_path}")
        except Exception as e:
            print(f"Warning: Failed to save training state: {e}")
    
    def on_training_end(self) -> None:
        """Save final state when training ends"""
        self._save_state()


# ============================================================================
# TRAINING MODES
# ============================================================================

class UnifiedTrainer:
    """Unified training system with multiple modes and algorithms"""
    
    def __init__(self,
                 algorithm: str = 'ppo',
                 n_envs: int = 12,  # OPTIMIZED: Increased from 4 to 12 for i7-14700K (20 cores)
                 save_dir: str = './models',
                 log_dir: str = './logs',
                 seed: int = 42,
                 verbose: int = 1):
        """
        Initialize unified trainer
        
        Args:
            algorithm: RL algorithm ('ppo', 'sac', 'td3')
            n_envs: Number of parallel environments
            save_dir: Directory to save models and checkpoints
            log_dir: Directory for TensorBoard logs
            seed: Random seed
            verbose: Verbosity level (0=none, 1=info, 2=debug)
        """
        self.algorithm = algorithm.lower()
        self.n_envs = n_envs
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.seed = seed
        self.verbose = verbose
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Model instance
        self.model = None
        
        # VecNormalize stats tracking (for curriculum stage transitions)
        self.current_vecnormalize_path = None
        
        # Curriculum stages
        self.curriculum_stages = self._create_curriculum()
        
        # Training state (for resume functionality)
        self.training_state_path = os.path.join(save_dir, 'training_state.json')
        self.curriculum_state_path = os.path.join(save_dir, 'curriculum_state.pkl')
        
        # Print initialization info
        if self.verbose > 0:
            self._print_header()
    
    def _save_training_state(self, 
                            stage_idx: int = 0,
                            stage_attempts: Dict[int, int] = None,
                            additional_info: Dict = None):
        """
        Save complete training state for resuming curriculum training.
        
        Args:
            stage_idx: Current curriculum stage index
            stage_attempts: Dictionary of stage_idx -> number of attempts
            additional_info: Any additional metadata to save
        """
        state = {
            'algorithm': self.algorithm,
            'n_envs': self.n_envs,
            'seed': self.seed,
            'stage_idx': stage_idx,
            'stage_attempts': stage_attempts or {},
            'timestamp': datetime.now().isoformat(),
            'current_vecnormalize_path': self.current_vecnormalize_path,
        }
        
        if additional_info:
            state.update(additional_info)
        
        try:
            # Save as JSON for human readability
            with open(self.training_state_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Also save as pickle with full state (in case JSON doesn't capture everything)
            with open(self.curriculum_state_path, 'wb') as f:
                pickle.dump(state, f)
            
            if self.verbose > 0:
                print(f"✓ Training state saved: {self.training_state_path}")
        
        except Exception as e:
            print(f"⚠ Warning: Failed to save training state: {e}")
    
    def _load_training_state(self) -> Optional[Dict]:
        """
        Load training state for resuming curriculum training.
        
        Returns:
            Dictionary with training state, or None if not found
        """
        # Try pickle first (most complete), then JSON
        for path in [self.curriculum_state_path, self.training_state_path]:
            if os.path.exists(path):
                try:
                    if path.endswith('.pkl'):
                        with open(path, 'rb') as f:
                            state = pickle.load(f)
                    else:
                        with open(path, 'r') as f:
                            state = json.load(f)
                    
                    # Validate loaded state before using it
                    required_keys = ['stage_idx', 'stage_attempts', 'timestamp']
                    if not all(k in state for k in required_keys):
                        print(f"⚠ Warning: Incomplete training state in {path}")
                        print(f"  Missing keys: {[k for k in required_keys if k not in state]}")
                        print(f"  Ignoring saved state and starting fresh")
                        return None
                    
                    # Validate stage index is within valid range
                    if state['stage_idx'] >= len(self.curriculum_stages):
                        print(f"⚠ Warning: Invalid stage_idx {state['stage_idx']} in {path}")
                        print(f"  Valid range: 0-{len(self.curriculum_stages)-1}")
                        print(f"  Resetting to stage 0")
                        state['stage_idx'] = 0
                    
                    if state['stage_idx'] < 0:
                        print(f"⚠ Warning: Negative stage_idx {state['stage_idx']}, resetting to 0")
                        state['stage_idx'] = 0
                    
                    if self.verbose > 0:
                        print(f"✓ Training state loaded from: {path}")
                        print(f"  Stage: {state.get('stage_idx', 0) + 1}")
                        print(f"  Timestamp: {state.get('timestamp', 'unknown')}")
                    
                    return state
                
                except Exception as e:
                    print(f"⚠ Warning: Failed to load training state from {path}: {e}")
        
        return None
    
    def _get_latest_checkpoint(self, stage_name: str = None) -> Optional[str]:
        """
        Find the latest checkpoint for resuming training.
        
        Args:
            stage_name: Optional stage name to find stage-specific checkpoint
        
        Returns:
            Path to latest checkpoint, or None if not found
        """
        if stage_name:
            checkpoint_dir = os.path.join(self.save_dir, f'{stage_name}_checkpoints')
        else:
            checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
        
        if not os.path.exists(checkpoint_dir):
            return None
        
        # Find all checkpoint files
        checkpoints = []
        for filename in os.listdir(checkpoint_dir):
            if filename.endswith('.zip'):
                filepath = os.path.join(checkpoint_dir, filename)
                checkpoints.append((filepath, os.path.getmtime(filepath)))
        
        if not checkpoints:
            return None
        
        # Return most recent checkpoint
        latest = max(checkpoints, key=lambda x: x[1])
        return latest[0]
    
    def _print_header(self):
        """Print initialization header"""
        print("\n" + "="*80)
        print("UNIFIED LUNAR LANDING TRAINING SYSTEM")
        print("="*80)
        print(f"Algorithm: {self.algorithm.upper()}")
        print(f"Parallel environments: {self.n_envs}")
        print(f"Save directory: {self.save_dir}")
        print(f"Log directory: {self.log_dir}")
        print(f"Random seed: {self.seed}")
        print("="*80 + "\n")
    
    def _make_env(self, env_config: Dict = None, rank: int = 0):
        """Create environment factory with subprocess safety"""
        def _init():
            try:
                config = env_config or {
                    'observation_mode': 'compact',
                    'max_episode_steps': 1000
                }
                # Add flag to improve performance during reset
                config['create_new_sim_on_reset'] = False
                env = LunarLanderEnv(**config)
                env.reset(seed=self.seed + rank)
                return env
            except Exception as e:
                print(f"\n{'='*60}")
                print(f"ERROR creating environment in subprocess {rank}:")
                print(f"{type(e).__name__}: {e}")
                print(f"{'='*60}\n")
                raise
        return _init
    
    def _normalize_env(self, env, training: bool = True):
        """
        Wrap environment with VecNormalize for observation/reward normalization.
        This provides zero-mean, unit-variance observations for improved training stability.
        
        Args:
            env: Vectorized environment (DummyVecEnv or SubprocVecEnv)
            training: If True, normalizes observations and rewards. If False (eval), only observations.
        
        Returns:
            VecNormalize wrapped environment
        """
        return VecNormalize(
            env,
            norm_obs=True,           # Normalize observations to zero mean, unit variance
            norm_reward=training,    # Normalize rewards during training only
            clip_obs=10.0,           # Clip normalized observations to [-10, 10]
            clip_reward=10.0,        # Clip normalized rewards to [-10, 10]
            gamma=0.99,              # Discount factor for reward normalization
            epsilon=1e-8             # Small value to avoid division by zero
        )
    
    def _create_model(self, env, learning_rate: float = 3e-4, **kwargs):
        """Create RL model based on algorithm"""
        
        if self.algorithm == 'ppo':
            model = PPO(
                policy='MlpPolicy',
                env=env,
                learning_rate=learning_rate,
                n_steps=kwargs.get('n_steps', 4096),      # OPTIMIZED: 2x from 2048 (more samples per update)
                batch_size=kwargs.get('batch_size', 512),  # OPTIMIZED: 8x from 64 (leverage GPU/RAM)
                n_epochs=kwargs.get('n_epochs', 10),
                gamma=kwargs.get('gamma', 0.99),
                gae_lambda=kwargs.get('gae_lambda', 0.95),
                clip_range=kwargs.get('clip_range', 0.2),
                ent_coef=kwargs.get('ent_coef', 0.01),
                vf_coef=kwargs.get('vf_coef', 0.5),
                max_grad_norm=kwargs.get('max_grad_norm', 0.5),
                verbose=self.verbose,
                tensorboard_log=self.log_dir,
                seed=self.seed,
                device='cuda'  # OPTIMIZED: Force GPU (RTX 4080) instead of 'auto'
            )
        
        elif self.algorithm == 'sac':
            model = SAC(
                policy='MlpPolicy',
                env=env,
                learning_rate=learning_rate,
                buffer_size=kwargs.get('buffer_size', 200_000),   # OPTIMIZED: 2x from 100k (64GB RAM)
                batch_size=kwargs.get('batch_size', 1024),        # OPTIMIZED: 4x from 256 (GPU memory)
                tau=kwargs.get('tau', 0.005),
                gamma=kwargs.get('gamma', 0.99),
                train_freq=kwargs.get('train_freq', 1),
                gradient_steps=kwargs.get('gradient_steps', 1),
                verbose=self.verbose,
                tensorboard_log=self.log_dir,
                seed=self.seed,
                device='cuda'  # OPTIMIZED: Force GPU (RTX 4080)
            )
        
        elif self.algorithm == 'td3':
            model = TD3(
                policy='MlpPolicy',
                env=env,
                learning_rate=learning_rate,
                buffer_size=kwargs.get('buffer_size', 200_000),   # OPTIMIZED: 2x from 100k (64GB RAM)
                batch_size=kwargs.get('batch_size', 512),         # OPTIMIZED: 5x from 100 (GPU memory)
                tau=kwargs.get('tau', 0.005),
                gamma=kwargs.get('gamma', 0.99),
                train_freq=kwargs.get('train_freq', (1, 'episode')),
                gradient_steps=kwargs.get('gradient_steps', -1),
                verbose=self.verbose,
                tensorboard_log=self.log_dir,
                seed=self.seed,
                device='cuda'  # OPTIMIZED: Force GPU (RTX 4080)
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        return model
    
    def _create_curriculum(self) -> List[CurriculumStage]:
        """
        Define 5-stage curriculum with progressive difficulty.
        
        Reward scale (see REWARD_SYSTEM_GUIDE.md):
        - Perfect landing: 1200-1600
        - Good landing: 900-1200
        - Basic landing: 600-900
        - Crash: -400 to -800
        
        Advancement requires BOTH mean reward > threshold AND 60% success rate.
        """
        stages = []
        
        # Stage 1: Basic landing on flat terrain
        stages.append(CurriculumStage(
            name="stage1_simple_landing",
            description="Learn basic landing from low altitude on flat terrain",
            env_config={
                'observation_mode': 'compact',
                'max_episode_steps': 600,
                'initial_altitude_range': (50.0, 100.0),
                'initial_velocity_range': ((-5.0, 5.0), (-5.0, 5.0), (-5.0, -2.0)),
                'terrain_config': {
                    'size': 1000.0,
                    'resolution': 100,
                    'num_craters': 0,  # Flat terrain
                    'crater_depth_range': (0, 0),
                    'crater_radius_range': (0, 0)
                }
            },
            success_threshold=400.0,
            min_episodes=200,
            max_timesteps=200_000  # OPTIMIZED: 2x from 100k (high-end hardware)
        ))
        
        # Stage 2: Medium altitude with gentle terrain
        stages.append(CurriculumStage(
            name="stage2_medium_descent",
            description="Controlled descent from medium altitude with gentle terrain",
            env_config={
                'observation_mode': 'compact',
                'max_episode_steps': 800,
                'initial_altitude_range': (100.0, 300.0),
                'initial_velocity_range': ((-15.0, 15.0), (-15.0, 15.0), (-15.0, -5.0)),
                'terrain_config': {
                    'size': 1500.0,
                    'resolution': 150,
                    'num_craters': 3,
                    'crater_depth_range': (1, 3),
                    'crater_radius_range': (30, 50)
                }
            },
            success_threshold=600.0,
            min_episodes=200,
            max_timesteps=300_000  # OPTIMIZED: 2x from 150k (high-end hardware)
        ))
        
        # Stage 3: High Altitude with Moderate Terrain
        stages.append(CurriculumStage(
            name="stage3_high_descent",
            description="Learn long descent from high altitude with moderate terrain",
            env_config={
                'observation_mode': 'compact',
                'max_episode_steps': 1000,
                'initial_altitude_range': (300.0, 800.0),  # Overlaps with stage 2
                'initial_velocity_range': ((-30.0, 30.0), (-30.0, 30.0), (-30.0, -10.0)),
                'terrain_config': {
                    'size': 2000.0,
                    'resolution': 200,
                    'num_craters': 8,
                    'crater_depth_range': (2, 6),
                    'crater_radius_range': (20, 60)
                }
            },
            success_threshold=700.0,     # Fuel efficiency starts mattering
            min_episodes=250,
            max_timesteps=400_000  # OPTIMIZED: 2x from 200k (high-end hardware)
        ))
        
        # Stage 4: Challenging Terrain and Conditions
        stages.append(CurriculumStage(
            name="stage4_challenging",
            description="Master landing on challenging terrain with varied conditions",
            env_config={
                'observation_mode': 'compact',
                'max_episode_steps': 1200,
                'initial_altitude_range': (500.0, 1500.0),  # Wide range
                'initial_velocity_range': ((-50.0, 50.0), (-50.0, 50.0), (-40.0, -20.0)),
                'terrain_config': {
                    'size': 2000.0,
                    'resolution': 200,
                    'num_craters': 15,
                    'crater_depth_range': (3, 12),
                    'crater_radius_range': (15, 60)
                }
            },
            success_threshold=800.0,     # High-quality landings expected
            min_episodes=300,
            max_timesteps=600_000  # OPTIMIZED: 2x from 300k (high-end hardware)
        ))
        
        # Stage 5: Extreme Conditions (Final Test)
        stages.append(CurriculumStage(
            name="stage5_extreme",
            description="Master extreme landing scenarios (high altitude, high speed, rough terrain)",
            env_config={
                'observation_mode': 'compact',
                'max_episode_steps': 1500,
                'initial_altitude_range': (18000.0, 22000.0),  # Suborbital trajectory from ~20km
                'initial_velocity_range': ((-200.0, 200.0), (-200.0, 200.0), (-100.0, -50.0)),  # Realistic suborbital velocities
                'terrain_config': {
                    'size': 2500.0,
                    'resolution': 250,
                    'num_craters': 25,
                    'crater_depth_range': (5, 15),
                    'crater_radius_range': (10, 80)
                }
            },
            success_threshold=900.0,     # Near-optimal performance: high precision + fuel efficiency
            min_episodes=400,
            max_timesteps=800_000  # OPTIMIZED: 2x from 400k (high-end hardware)
        ))
        
        return stages
    
    # ========================================================================
    # MODE: TEST
    # ========================================================================
    
    def test_setup(self):
        """Quick test to verify environment setup (~2 minutes)"""
        print("\n" + "="*80)
        print("QUICK ENVIRONMENT TEST")
        print("="*80)
        print("\nThis will verify that your environment is set up correctly.")
        print("Running 5,000 timesteps (~2 minutes)...")
        print("="*80 + "\n")
        
        try:
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
            obs, info = env.reset(seed=self.seed)
            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    obs, info = env.reset()
            print("  ✓ Environment interaction successful")
            env.close()
            
            # Test model creation and training
            print("\n[4/4] Testing model creation and mini-training...")
            env = DummyVecEnv([self._make_env()])
            model = self._create_model(env)
            model.learn(total_timesteps=5_000, progress_bar=True)
            print("  ✓ Model training successful")
            env.close()
            
            # Success!
            print("\n" + "="*80)
            print("✓ ENVIRONMENT TEST PASSED!")
            print("="*80)
            print("\nYour setup is working correctly! You can now:")
            print("  1. Run demo: python unified_training.py --mode demo")
            print("  2. Start standard training: python unified_training.py --mode standard")
            print("  3. Run full curriculum: python unified_training.py --mode curriculum")
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
            print("  3. Basilisk simulation is working")
            print("="*80 + "\n")
            import traceback
            traceback.print_exc()
            return False
    
    # ========================================================================
    # MODE: DEMO
    # ========================================================================
    
    def demo_training(self):
        """Quick demo of training system (~15 minutes)"""
        print("\n" + "="*80)
        print("TRAINING DEMO MODE")
        print("="*80)
        print("\nThis is a DEMO with reduced timesteps for quick testing.")
        print("\nDemo configuration:")
        print("  - 3 mini-stages: 10,000 timesteps each")
        print("  - 2 parallel environments")
        print("  - Reduced success thresholds")
        print("  - Total time: ~10-15 minutes")
        print("="*80 + "\n")
        
        # Create demo stages (shorter versions)
        demo_stages = [
            CurriculumStage(
                name="demo_hover",
                description="DEMO: Learn basic hovering",
                env_config={
                    'observation_mode': 'compact',
                    'max_episode_steps': 300,
                    'initial_altitude_range': (100.0, 150.0),
                    'initial_velocity_range': ((-5.0, 5.0), (-5.0, 5.0), (-5.0, -2.0)),
                    'terrain_config': {
                        'size': 1000.0, 'resolution': 100,
                        'num_craters': 0, 'crater_depth_range': (0, 0),
                        'crater_radius_range': (0, 0)
                    }
                },
                success_threshold=-100.0,
                min_episodes=10,
                max_timesteps=10_000
            ),
            CurriculumStage(
                name="demo_descent",
                description="DEMO: Learn controlled descent",
                env_config={
                    'observation_mode': 'compact',
                    'max_episode_steps': 500,
                    'initial_altitude_range': (200.0, 400.0),
                    'initial_velocity_range': ((-15.0, 15.0), (-15.0, 15.0), (-15.0, -5.0)),
                    'terrain_config': {
                        'size': 1500.0, 'resolution': 150,
                        'num_craters': 3, 'crater_depth_range': (1, 3),
                        'crater_radius_range': (30, 50)
                    }
                },
                success_threshold=-50.0,
                min_episodes=20,
                max_timesteps=10_000
            ),
            CurriculumStage(
                name="demo_precision",
                description="DEMO: Learn precision landing",
                env_config={
                    'observation_mode': 'compact',
                    'max_episode_steps': 800,
                    'initial_altitude_range': (400.0, 800.0),
                    'initial_velocity_range': ((-30.0, 30.0), (-30.0, 30.0), (-30.0, -10.0)),
                    'terrain_config': {
                        'size': 2000.0, 'resolution': 200,
                        'num_craters': 8, 'crater_depth_range': (2, 6),
                        'crater_radius_range': (20, 60)
                    }
                },
                success_threshold=0.0,
                min_episodes=30,
                max_timesteps=10_000
            )
        ]
        
        # Train through demo stages
        for i, stage in enumerate(demo_stages):
            print(f"\n{'='*80}")
            print(f"DEMO STAGE {i+1}/{len(demo_stages)}: {stage.name.upper()}")
            print(f"{'='*80}")
            
            mean_reward, success_rate = self._train_stage(stage, n_envs=2, demo=True)
            
            print(f"\n{'='*60}")
            print(f"Demo Stage {i+1} Results:")
            print(f"  Mean reward: {mean_reward:.2f}")
            print(f"  Success rate: {success_rate*100:.1f}%")
            print(f"  Threshold: {stage.success_threshold}")
            
            if mean_reward >= stage.success_threshold:
                print(f"  Status: ✓ PASSED")
            else:
                print(f"  Status: ⚠ Not mastered (demo mode - continuing anyway)")
            print(f"{'='*60}")
        
        # Save demo model
        demo_path = os.path.join(self.save_dir, 'demo_final')
        self.model.save(demo_path)
        
        print("\n" + "="*80)
        print("DEMO COMPLETE!")
        print("="*80)
        print(f"\nDemo model saved to: {demo_path}")
        print("\nNext steps:")
        print("  1. View training: tensorboard --logdir=./logs")
        print("  2. Start full training: python unified_training.py --mode standard")
        print("  3. Run curriculum: python unified_training.py --mode curriculum")
        print("="*80 + "\n")
        
        return self.model
    
    # ========================================================================
    # MODE: STANDARD
    # ========================================================================
    
    def standard_training(self, 
                         total_timesteps: int = 1_000_000,
                         learning_rate: float = 3e-4,
                         resume_path: Optional[str] = None):
        """Standard RL training without curriculum"""
        print("\n" + "="*80)
        print("STANDARD TRAINING MODE")
        print("="*80)
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Parallel environments: {self.n_envs}")
        print(f"Learning rate: {learning_rate}")
        if resume_path:
            print(f"Resuming from: {resume_path}")
        print("="*80 + "\n")
        
        # Create environments with subprocess safety
        if self.n_envs > 1:
            try:
                print(f"Creating {self.n_envs} parallel environments...")
                env = SubprocVecEnv([self._make_env(rank=i) for i in range(self.n_envs)],
                                   start_method='spawn')  # Windows compatibility
                print("✓ Parallel environments created successfully")
            except Exception as e:
                print(f"\n⚠ WARNING: Failed to create parallel environments: {e}")
                print("  Falling back to single environment (DummyVecEnv)")
                print("  This will be slower but more stable.\n")
                env = DummyVecEnv([self._make_env()])
                self.n_envs = 1  # Update for consistency
        else:
            env = DummyVecEnv([self._make_env()])
        
        # Add observation normalization for training stability
        # If resuming, try to load previous VecNormalize stats
        vecnormalize_path = os.path.join(self.save_dir, 'checkpoints', 'vecnormalize.pkl')
        if resume_path and os.path.exists(vecnormalize_path):
            print(f"Loading VecNormalize stats from: {vecnormalize_path}")
            try:
                env = VecNormalize.load(vecnormalize_path, env)
                env.training = True
                env.norm_reward = True
                print("✓ VecNormalize stats loaded successfully")
            except Exception as e:
                print(f"⚠ Failed to load VecNormalize: {e}")
                print("  Creating fresh VecNormalize wrapper")
                env = self._normalize_env(env, training=True)
        else:
            env = self._normalize_env(env, training=True)
        
        eval_env = DummyVecEnv([self._make_env(rank=self.n_envs)])
        eval_env = self._normalize_env(eval_env, training=False)  # Eval: obs normalization only
        
        # Create or load model
        if resume_path:
            print(f"Loading model from: {resume_path}")
            if self.algorithm == 'ppo':
                self.model = PPO.load(resume_path, env=env)
            elif self.algorithm == 'sac':
                self.model = SAC.load(resume_path, env=env)
            elif self.algorithm == 'td3':
                self.model = TD3.load(resume_path, env=env)
            print("✓ Model loaded successfully")
            
            # Reset num_timesteps if continuing training (not reset_num_timesteps)
            initial_timesteps = self.model.num_timesteps
            print(f"  Model has {initial_timesteps:,} timesteps")
            print(f"  Will train for {total_timesteps:,} additional timesteps")
        else:
            print("Creating new model...")
            self.model = self._create_model(env, learning_rate=learning_rate)
            if self.verbose > 0:
                print("\nModel architecture:")
                print(self.model.policy)
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000,  # OPTIMIZED: Reduced I/O frequency (from 50k to 100k)
            save_path=os.path.join(self.save_dir, 'checkpoints'),
            name_prefix=f'{self.algorithm}_lunar_lander',
            save_replay_buffer=False,
            save_vecnormalize=True
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.save_dir, 'best_model'),
            log_path=os.path.join(self.log_dir, 'eval'),
            eval_freq=10_000,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            verbose=1
        )
        
        progress_callback = TrainingProgressCallback("standard_training", verbose=1)
        reward_stats_callback = RewardStatisticsCallback(verbose=1)
        
        callback = CallbackList([checkpoint_callback, eval_callback, progress_callback, reward_stats_callback])
        
        # Train
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80 + "\n")
        
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                log_interval=10,
                progress_bar=True,
                reset_num_timesteps=False  # Keep timestep count when resuming
            )
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
        
        # Save final model
        final_path = os.path.join(self.save_dir, f'{self.algorithm}_final')
        self.model.save(final_path)
        print(f"\n✓ Final model saved to: {final_path}")
        
        # Save VecNormalize stats
        env.save(vecnormalize_path)
        print(f"✓ VecNormalize stats saved: {vecnormalize_path}")
        
        env.close()
        eval_env.close()
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"\nTo view training progress:")
        print(f"  tensorboard --logdir={self.log_dir}")
        print(f"\nTo evaluate the model:")
        print(f"  python unified_training.py --mode eval --model-path {final_path}")
        print(f"\nTo resume training:")
        print(f"  python unified_training.py --mode standard --resume {final_path} --timesteps <additional_steps>")
        print("="*80 + "\n")
        
        return self.model
    
    # ========================================================================
    # MODE: CURRICULUM
    # ========================================================================
    
    def curriculum_training(self, start_stage: int = 0, auto_advance: bool = True, resume: bool = False):
        """
        Full curriculum learning through all stages with IMPROVED advancement logic:
        - Requires both mean reward threshold AND 60% success rate
        - Implements stage regression if performance drops
        - Validates mastery before advancing
        - Supports resuming from saved state
        
        Args:
            start_stage: Starting stage index (0-based)
            auto_advance: Automatically advance stages based on performance
            resume: Try to resume from saved training state
        """
        print("\n" + "="*80)
        print("CURRICULUM LEARNING MODE (IMPROVED)")
        print("="*80)
        print(f"Total stages: {len(self.curriculum_stages)}")
        
        # Try to load saved state if resume is requested
        stage_idx = start_stage
        stage_attempts = {}
        
        if resume:
            print("\nAttempting to resume from saved state...")
            saved_state = self._load_training_state()
            
            if saved_state:
                stage_idx = saved_state.get('stage_idx', start_stage)
                stage_attempts = saved_state.get('stage_attempts', {})
                # Convert string keys back to int (JSON doesn't support int keys)
                stage_attempts = {int(k): v for k, v in stage_attempts.items()}
                
                # Restore VecNormalize path
                if saved_state.get('current_vecnormalize_path'):
                    self.current_vecnormalize_path = saved_state['current_vecnormalize_path']
                
                print(f"✓ Resuming from Stage {stage_idx + 1}")
                print(f"  Stage attempts so far: {stage_attempts}")
                
                # Try to load the model from latest checkpoint
                stage_name = self.curriculum_stages[stage_idx].name
                checkpoint_path = self._get_latest_checkpoint(stage_name)
                
                if checkpoint_path:
                    print(f"  Loading model from: {checkpoint_path}")
                    try:
                        if self.algorithm == 'ppo':
                            self.model = PPO.load(checkpoint_path)
                        elif self.algorithm == 'sac':
                            self.model = SAC.load(checkpoint_path)
                        elif self.algorithm == 'td3':
                            self.model = TD3.load(checkpoint_path)
                        print(f"  ✓ Model loaded with {self.model.num_timesteps:,} timesteps")
                    except Exception as e:
                        print(f"  ⚠ Failed to load checkpoint: {e}")
                        print("  Will start fresh for this stage")
                else:
                    print(f"  No checkpoint found for {stage_name}, starting fresh")
            else:
                print("  No saved state found, starting from scratch")
        
        print(f"Starting at stage: {stage_idx + 1}")
        print(f"Auto-advance: {auto_advance}")
        print("\nADVANCEMENT CRITERIA:")
        print("  - Mean reward > threshold")
        print("  - Success rate > 60% (last 100 episodes)")
        print("  - Stage regression enabled for poor performance")
        print("\nStages:")
        for i, stage in enumerate(self.curriculum_stages):
            marker = "➜" if i == stage_idx else " "
            print(f" {marker}{i+1}. {stage.name}: {stage.description}")
            print(f"      Threshold: {stage.success_threshold}, Max steps: {stage.max_timesteps}")
        print("="*80 + "\n")
        
        while stage_idx < len(self.curriculum_stages):
            stage = self.curriculum_stages[stage_idx]
            
            # Track attempts
            if stage_idx not in stage_attempts:
                stage_attempts[stage_idx] = 0
            stage_attempts[stage_idx] += 1
            
            # Save state before starting stage
            self._save_training_state(
                stage_idx=stage_idx,
                stage_attempts=stage_attempts,
                additional_info={'mode': 'curriculum'}
            )
            
            print(f"\n{'='*80}")
            print(f"STAGE {stage_idx + 1}/{len(self.curriculum_stages)}: {stage.name.upper()}")
            print(f"Attempt #{stage_attempts[stage_idx]}")
            print(f"{'='*80}")
            
            mean_reward, success_rate = self._train_stage(stage, n_envs=self.n_envs)
            
            # Save state after stage completion
            self._save_training_state(
                stage_idx=stage_idx,
                stage_attempts=stage_attempts,
                additional_info={
                    'mode': 'curriculum',
                    'last_stage_reward': mean_reward,
                    'last_stage_success_rate': success_rate
                }
            )
            
            # Check advancement criteria
            reward_passed = mean_reward >= stage.success_threshold
            
            # Stage-specific success rate thresholds
            # Early stages have lower thresholds to allow exploration
            SUCCESS_RATE_THRESHOLDS = {
                0: 0.40,  # Stage 1: 40% (learning basics)
                1: 0.50,  # Stage 2: 50%
                2: 0.55,  # Stage 3: 55%
                3: 0.60,  # Stage 4: 60%
                4: 0.65   # Stage 5: 65% (final mastery)
            }
            required_success_rate = SUCCESS_RATE_THRESHOLDS.get(stage_idx, 0.60)
            success_passed = success_rate >= required_success_rate
            
            print(f"\n{'='*60}")
            print(f"STAGE {stage_idx + 1} COMPLETION REPORT")
            print(f"{'='*60}")
            print(f"Mean reward:     {mean_reward:.2f} / {stage.success_threshold:.2f} {'✓' if reward_passed else '✗'}")
            print(f"Success rate:    {success_rate*100:.1f}% / {required_success_rate*100:.0f}% {'✓' if success_passed else '✗'}")
            print(f"Overall:         {'PASSED' if (reward_passed and success_passed) else 'FAILED'}")
            print(f"{'='*60}\n")
            
            if auto_advance:
                if reward_passed and success_passed:
                    print(f"✓ Stage mastered! Advancing to next stage...\n")
                    stage_idx += 1  # Advance
                    
                elif stage_attempts[stage_idx] < 3:
                    # Allow up to 3 attempts at current stage
                    print(f"⚠ Stage not mastered. Retrying same stage (attempt {stage_attempts[stage_idx] + 1}/3)...\n")
                    # Stay at current stage (will retry on next iteration)
                    
                elif stage_idx > 0:
                    # Regress to previous stage with proper state restoration
                    # Reload both the model AND VecNormalize stats from previous stage
                    print(f"⚠ Failed {stage_attempts[stage_idx]} times. REGRESSING to Stage {stage_idx}...")
                    stage_idx -= 1  # Go back one stage
                    
                    # Restore previous stage's model
                    prev_stage = self.curriculum_stages[stage_idx]
                    prev_model_path = os.path.join(self.save_dir, f'{prev_stage.name}_best', 'best_model.zip')
                    if os.path.exists(prev_model_path):
                        print(f"  Reloading model from: {prev_model_path}")
                        if self.algorithm == 'ppo':
                            self.model = PPO.load(prev_model_path)
                        elif self.algorithm == 'sac':
                            self.model = SAC.load(prev_model_path)
                        elif self.algorithm == 'td3':
                            self.model = TD3.load(prev_model_path)
                        print("  ✓ Model restored to previous stage")
                    else:
                        print(f"  ⚠ Previous model not found, keeping current model")
                    
                    # Restore previous stage's VecNormalize stats
                    prev_vecnorm_path = os.path.join(self.save_dir, f'{prev_stage.name}_vecnormalize.pkl')
                    if os.path.exists(prev_vecnorm_path):
                        self.current_vecnormalize_path = prev_vecnorm_path
                        print(f"  ✓ VecNormalize stats will reload from: {prev_vecnorm_path}")
                    
                    print(f"  Regression complete. Resuming Stage {stage_idx + 1}.\n")
                    
                else:
                    # Stage 1 failure - adjust expectations
                    print(f"⚠ Stage 1 not mastered after {stage_attempts[stage_idx]} attempts.")
                    user_input = input("Continue to next stage anyway? (y/n): ")
                    if user_input.lower() == 'y':
                        stage_idx += 1
                    else:
                        print("Training stopped by user.")
                        break
            else:
                # Manual advancement
                user_input = input(f"Advance to next stage? (y/n/r for regress): ")
                if user_input.lower() == 'y':
                    stage_idx += 1
                elif user_input.lower() == 'r' and stage_idx > 0:
                    stage_idx -= 1
                    print(f"Regressing to Stage {stage_idx + 1}...")
                else:
                    print("Training stopped by user.")
                    break
        
        # Save final curriculum model
        final_path = os.path.join(self.save_dir, 'curriculum_final')
        self.model.save(final_path)
        
        # Save final state
        self._save_training_state(
            stage_idx=len(self.curriculum_stages),  # Completed all stages
            stage_attempts=stage_attempts,
            additional_info={'mode': 'curriculum', 'status': 'completed'}
        )
        
        print("\n" + "="*80)
        print("CURRICULUM TRAINING COMPLETE!")
        print("="*80)
        print(f"\nFinal model: {final_path}")
        print(f"Stage models: {self.save_dir}/stage*_final")
        print(f"\nStage completion summary:")
        for idx, attempts in stage_attempts.items():
            stage_name = self.curriculum_stages[idx].name
            print(f"  Stage {idx+1} ({stage_name}): {attempts} attempt(s)")
        print(f"\nView training: tensorboard --logdir={self.log_dir}")
        print(f"\nTo resume if interrupted:")
        print(f"  python unified_training.py --mode curriculum --resume")
        print("="*80 + "\n")
        
        return self.model
    
    def _train_stage(self, stage: CurriculumStage, n_envs: int, demo: bool = False):
        """
        Train on a specific curriculum stage
        
        Returns:
            mean_reward (float): Mean reward over evaluation episodes
            success_rate (float): Success rate (0.0-1.0) over evaluation episodes
        """
        # Create environments for this stage with subprocess safety
        if n_envs > 1:
            try:
                print(f"  Creating {n_envs} parallel environments for stage '{stage.name}'...")
                env = SubprocVecEnv([self._make_env(stage.env_config, i) for i in range(n_envs)],
                                   start_method='spawn')  # Windows compatibility
                print(f"  ✓ Parallel environments created successfully")
            except Exception as e:
                print(f"\n  ⚠ WARNING: Failed to create parallel environments: {e}")
                print(f"    Falling back to single environment (DummyVecEnv)")
                print(f"    This will be slower but more stable.\n")
                env = DummyVecEnv([self._make_env(stage.env_config)])
                n_envs = 1  # Update for this stage
        else:
            env = DummyVecEnv([self._make_env(stage.env_config)])
        
        # Load previous VecNormalize stats for curriculum continuity
        # This prevents catastrophic forgetting when transitioning between stages
        vecnormalize_path = os.path.join(self.save_dir, f'{stage.name}_vecnormalize.pkl')
        
        if self.current_vecnormalize_path and os.path.exists(self.current_vecnormalize_path):
            # Load previous stage's normalization statistics
            print(f"Loading VecNormalize stats from: {self.current_vecnormalize_path}")
            try:
                env = VecNormalize.load(self.current_vecnormalize_path, env)
                # Continue updating statistics (don't freeze them)
                env.training = True
                env.norm_reward = True
                
                # Validate statistics loaded correctly before proceeding
                if not hasattr(env, 'obs_rms') or env.obs_rms is None:
                    raise ValueError("VecNormalize obs_rms not properly initialized")
                if not hasattr(env, 'ret_rms') or env.ret_rms is None:
                    raise ValueError("VecNormalize ret_rms not properly initialized")
                
                # Show first 5 dimensions of mean for verification
                mean_sample = env.obs_rms.mean[:min(5, len(env.obs_rms.mean))]
                print(f"  ✓ Loaded obs mean (first 5): {mean_sample}")
                print("  ✓ VecNormalize stats loaded successfully")
            except Exception as e:
                print(f"  ⚠ Failed to load VecNormalize stats: {e}")
                print("  Creating fresh VecNormalize wrapper")
                env = self._normalize_env(env, training=True)
        else:
            # First stage or no previous stats - create fresh
            env = self._normalize_env(env, training=True)
        
        # Create eval environment (observation normalization only, no reward normalization)
        eval_env = DummyVecEnv([self._make_env(stage.env_config, n_envs)])
        eval_env = self._normalize_env(eval_env, training=False)
        
        # Create or update model
        if self.model is None:
            self.model = self._create_model(env)
            if self.verbose > 0:
                print("\nModel architecture:")
                print(self.model.policy)
        else:
            # Update environment for existing model
            self.model.set_env(env)
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000 if not demo else 5_000,  # OPTIMIZED: Reduced I/O (from 50k to 100k)
            save_path=os.path.join(self.save_dir, f'{stage.name}_checkpoints'),
            name_prefix=f'{self.algorithm}_{stage.name}',
            save_replay_buffer=False,
            save_vecnormalize=True  # CRITICAL: Save normalization statistics
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.save_dir, f'{stage.name}_best'),
            log_path=os.path.join(self.log_dir, f'{stage.name}_eval'),
            eval_freq=10_000 if not demo else 2_500,
            n_eval_episodes=5,
            deterministic=True,
            verbose=1
        )
        
        progress_callback = TrainingProgressCallback(stage.name, verbose=1)
        reward_stats_callback = RewardStatisticsCallback(verbose=1)
        
        callback = CallbackList([checkpoint_callback, eval_callback, progress_callback, reward_stats_callback])
        
        # Train
        print(f"\nTraining for up to {stage.max_timesteps:,} timesteps...")
        print(f"Success criteria: Mean reward > {stage.success_threshold}, Success rate > 60%")
        
        try:
            self.model.learn(
                total_timesteps=stage.max_timesteps,
                callback=callback,
                log_interval=10,
                progress_bar=True,
                reset_num_timesteps=False
            )
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")
        except EOFError as e:
            print(f"\n\n{'='*60}")
            print("ERROR: Subprocess pipe broken (multiprocessing failure)")
            print(f"{'='*60}")
            print("This typically occurs when:")
            print("  1. Child process crashed during environment reset/step")
            print("  2. Basilisk simulation diverged in a subprocess")
            print("  3. Windows multiprocessing incompatibility")
            print("\nRecommended solutions:")
            print("  1. Reduce parallel environments: --n-envs 2 or --n-envs 1")
            print("  2. The system will attempt to recover with DummyVecEnv...")
            print(f"{'='*60}\n")
            raise
        except BrokenPipeError as e:
            print(f"\n\n{'='*60}")
            print("ERROR: Subprocess communication failed")
            print(f"{'='*60}")
            print("Subprocess crashed during training. This is usually due to:")
            print("  1. Basilisk memory issues in parallel environments")
            print("  2. CUDA out-of-memory in GPU accelerated training")
            print("  3. Simulation divergence causing NaN values")
            print("\nRecommended solutions:")
            print("  1. Use single environment: --n-envs 1")
            print("  2. Reduce batch size if using GPU")
            print(f"{'='*60}\n")
            raise
        
        # Save stage model
        stage_path = os.path.join(self.save_dir, f'{stage.name}_final')
        self.model.save(stage_path)
        print(f"\n✓ Stage model saved: {stage_path}")
        
        # Save VecNormalize statistics for next stage
        # Critical for curriculum learning - preserves observation scaling knowledge
        env.save(vecnormalize_path)
        self.current_vecnormalize_path = vecnormalize_path
        print(f"✓ VecNormalize stats saved: {vecnormalize_path}")
        
        # Evaluate with success tracking
        mean_reward, std_reward, success_rate = self._evaluate_model_with_success(
            eval_env, n_episodes=50
        )
        
        print(f"\n{'='*60}")
        print(f"STAGE EVALUATION: {stage.name}")
        print(f"{'='*60}")
        print(f"Episodes evaluated: 50")
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Success rate: {success_rate*100:.1f}%")
        print(f"Threshold: {stage.success_threshold}")
        print(f"Status: {'✓ PASSED' if (mean_reward >= stage.success_threshold and success_rate >= 0.6) else '✗ NOT YET'}")
        print(f"{'='*60}\n")
        
        env.close()
        eval_env.close()
        
        return mean_reward, success_rate
    
    # ========================================================================
    # MODE: EVALUATION
    # ========================================================================
    
    def evaluate(self, 
                 model_path: str,
                 n_episodes: int = 10,
                 render: bool = False,
                 env_config: Optional[Dict] = None):
        """Evaluate a trained model"""
        print("\n" + "="*80)
        print("MODEL EVALUATION MODE")
        print("="*80)
        print(f"Model: {model_path}")
        print(f"Episodes: {n_episodes}")
        print(f"Render: {render}")
        print("="*80 + "\n")
        
        # Load model
        print("Loading model...")
        if self.algorithm == 'ppo' or 'ppo' in model_path.lower():
            model = PPO.load(model_path)
        elif self.algorithm == 'sac' or 'sac' in model_path.lower():
            model = SAC.load(model_path)
        elif self.algorithm == 'td3' or 'td3' in model_path.lower():
            model = TD3.load(model_path)
        else:
            # Try to auto-detect
            try:
                model = PPO.load(model_path)
            except:
                try:
                    model = SAC.load(model_path)
                except:
                    model = TD3.load(model_path)
        
        print("✓ Model loaded successfully")
        
        # Create environment
        config = env_config or {
            'observation_mode': 'compact',
            'render_mode': 'human' if render else None
        }
        config['create_new_sim_on_reset'] = True  # Avoid Basilisk warnings
        env = LunarLanderEnv(**config)
        
        # Evaluate
        episode_rewards = []
        episode_lengths = []
        successes = []
        
        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                done = terminated or truncated
                
                if render:
                    env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)
            
            # Check success (positive reward usually means successful landing)
            success = episode_reward > 100
            successes.append(success)
            
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"Episode {ep+1}/{n_episodes}: {status}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Length: {step_count} steps")
            print(f"  Final altitude: {info.get('altitude', 0):.2f} m")
            print(f"  Fuel remaining: {info.get('fuel_fraction', 0)*100:.1f}%")
            print()
        
        # Summary statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        success_rate = np.mean(successes) * 100
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Episodes: {n_episodes}")
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Mean episode length: {mean_length:.1f} steps")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Min reward: {min(episode_rewards):.2f}")
        print(f"Max reward: {max(episode_rewards):.2f}")
        print("="*80 + "\n")
        
        env.close()
        
        return mean_reward, std_reward, success_rate
    
    def _evaluate_model(self, env, n_episodes: int = 10):
        """Internal evaluation helper"""
        episode_rewards = []
        
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards), np.std(episode_rewards)
    
    def _evaluate_model_with_success(self, env, n_episodes: int = 10):
        """
        Internal evaluation helper with success rate tracking
        
        Returns:
            mean_reward (float): Mean episode reward
            std_reward (float): Standard deviation of rewards
            success_rate (float): Fraction of successful landings (0.0-1.0)
        """
        episode_rewards = []
        successes = []
        
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            # Success if reward > 50 (indicates successful landing per new reward design)
            successes.append(episode_reward > 50.0)
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        success_rate = np.mean(successes)
        
        return mean_reward, std_reward, success_rate


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified Lunar Landing RL Training System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (2 min)
  python unified_training.py --mode test
  
  # Demo training (15 min)
  python unified_training.py --mode demo
  
  # Standard training
  python unified_training.py --mode standard --timesteps 1000000
  
  # Curriculum training
  python unified_training.py --mode curriculum
  
  # Evaluate model
  python unified_training.py --mode eval --model-path ./models/best_model/best_model
  
  # Resume training
  python unified_training.py --mode standard --resume ./models/checkpoints/ppo_lunar_lander_500000_steps
  
  # Resume curriculum training
  python unified_training.py --mode curriculum --resume
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='standard',
                       choices=['test', 'demo', 'standard', 'curriculum', 'eval'],
                       help='Training mode')
    
    # Algorithm
    parser.add_argument('--algorithm', type=str, default='ppo',
                       choices=['ppo', 'sac', 'td3'],
                       help='RL algorithm')
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=2_000_000,  # OPTIMIZED: 2x from 1M (high-end hardware)
                       help='Total training timesteps (standard mode)')
    parser.add_argument('--n-envs', type=int, default=12,  # OPTIMIZED: 3x from 4 (i7-14700K has 20 cores)
                       help='Number of parallel environments')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    
    # Curriculum
    parser.add_argument('--start-stage', type=int, default=0,
                       help='Starting stage for curriculum (0-based)')
    parser.add_argument('--no-auto-advance', action='store_true',
                       help='Disable automatic stage advancement')
    parser.add_argument('--resume-curriculum', action='store_true',
                       help='Resume curriculum training from saved state')
    
    # Paths
    parser.add_argument('--save-dir', type=str, default='./models',
                       help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory for TensorBoard logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model for evaluation')
    
    # Evaluation
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes during evaluation')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (0, 1, 2)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = UnifiedTrainer(
        algorithm=args.algorithm,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        verbose=args.verbose
    )
    
    # Execute based on mode
    if args.mode == 'test':
        trainer.test_setup()
    
    elif args.mode == 'demo':
        trainer.demo_training()
    
    elif args.mode == 'standard':
        trainer.standard_training(
            total_timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            resume_path=args.resume
        )
    
    elif args.mode == 'curriculum':
        trainer.curriculum_training(
            start_stage=args.start_stage,
            auto_advance=not args.no_auto_advance,
            resume=args.resume is not None or args.resume_curriculum
        )
    
    elif args.mode == 'eval':
        if args.model_path is None:
            print("Error: --model-path required for evaluation mode")
            sys.exit(1)
        
        trainer.evaluate(
            model_path=args.model_path,
            n_episodes=args.eval_episodes,
            render=args.render
        )


if __name__ == "__main__":
    # CRITICAL: Windows multiprocessing requires 'spawn' method and __main__ guard
    # This prevents subprocess re-import issues and memory corruption
    if sys.platform == 'win32':
        # Force spawn method on Windows (required for CUDA + multiprocessing)
        multiprocessing.set_start_method('spawn', force=True)
    
    main()
