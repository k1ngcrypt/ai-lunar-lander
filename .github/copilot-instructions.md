# AI Lunar Lander - Copilot Instructions

## Project Overview
This is a reinforcement learning project for training AI agents to perform autonomous lunar landings using:
- **Basilisk**: High-fidelity astrodynamics simulation framework (6-DOF spacecraft dynamics)
- **Stable Baselines3**: PyTorch-based RL library (PPO, SAC, TD3 algorithms)
- **Gymnasium**: Environment interface for RL agents
- **Starship HLS**: Realistic lunar lander configuration (1.3M kg initial mass, 3 Raptor engines)

## Architecture & Component Boundaries

### Three-Layer Architecture
1. **Simulation Layer** (`ScenarioLunarLanderStarter.py`): Basilisk physics simulation
   - Spacecraft dynamics with fuel depletion (`FuelTank` effectors for CH4/LOX)
   - Lunar gravity field and analytical terrain model (`LunarRegolithModel`)
   - Sensor systems: IMU, LIDAR (64-ray cone scan), fuel gauges
   - Thruster controller mapping actions to 3 Raptor engines + RCS

2. **Environment Layer** (`lunar_lander_env.py`): Gymnasium wrapper
   - Translates Basilisk simulation into Gym API (`reset()`, `step()`, render)
   - **Observation space**: 23D compact (position, velocity, attitude quaternion, IMU, LIDAR stats, fuel) or 200+ full
   - **Action space**: 4D compact (main throttle 0.4-1.0, pitch/yaw/roll torques ±1) or 9D full
   - **Reward shaping**: Composite function balancing altitude descent, fuel efficiency, landing success (±1000 bonus/penalty)
   - **Termination**: Success (altitude < 5m, velocity < 3 m/s, upright), crash, or timeout

3. **Training Layer** (`unified_training.py`): RL orchestration
   - **Curriculum learning**: 5 progressive stages (hover → simple descent → precision landing → challenging terrain → extreme conditions)
   - Multi-algorithm support (PPO default, SAC for sample efficiency, TD3 for deterministic control)
   - Parallel environments (`SubprocVecEnv` with `--n-envs 4-16`)
   - Checkpointing every 50k steps, best model auto-save via `EvalCallback`

### Key Integration Points
- **Basilisk ↔ Gymnasium**: `LunarLanderEnv._create_simulation()` initializes Basilisk sim, updates via `scSim.ExecuteSimulation()` at 0.1s timestep
- **Curriculum advancement**: `TrainingProgressCallback` logs stage info to TensorBoard; auto-advance when mean reward > threshold for min_episodes
- **Terrain generation**: `generate_terrain.py` creates procedural lunar surfaces via Gaussian craters + noise; loaded by `LunarRegolithModel.load_terrain_from_file()`

## Critical Developer Workflows

### Setup & Dependencies
```powershell
# Basilisk is pre-built in ./basilisk/dist3/ (no rebuild needed)
# Install Python dependencies
pip install stable-baselines3[extra] gymnasium numpy matplotlib tensorboard
```

### Training Commands (Primary Workflow)
```powershell
# 1. ALWAYS test setup first (2 min sanity check)
python unified_training.py --mode test

# 2. Curriculum training (recommended, 4-8 hrs)
python unified_training.py --mode curriculum --n-envs 4

# 3. Monitor via TensorBoard (separate terminal)
tensorboard --logdir=./logs

# 4. Evaluate trained model
python unified_training.py --mode eval --model-path ./models/curriculum_final --eval-episodes 20
```

### Quick Development Iteration
```powershell
# Fast demo for code changes (15 min)
python unified_training.py --mode demo

# Single-stage testing (avoid full curriculum)
python unified_training.py --mode standard --timesteps 100000
```

### Debugging Environment Issues
```python
# Standalone environment test (bottom of lunar_lander_env.py)
python lunar_lander_env.py

# Check Gymnasium API compliance
from stable_baselines3.common.env_checker import check_env
env = LunarLanderEnv()
check_env(env)
```

## Project-Specific Conventions

### Reward Function Philosophy (Critical for Tuning)
**Located in**: `lunar_lander_env.py::_compute_reward()`
- **Dense shaping**: Small penalties per step to guide learning (altitude descent -0.001*h, velocity penalties -0.1*|v|)
- **Sparse bonuses**: Large terminal rewards for success (+1000) or crash (-500)
- **Efficiency incentives**: Fuel conservation (+0.001*fuel_fraction), precision bonus (up to +100 for landing within 10m of target)
- **DO NOT** add generic "exploration bonuses" - terrain randomization provides sufficient diversity

### Curriculum Stage Design Pattern
**Located in**: `unified_training.py::_create_curriculum()`
Each `CurriculumStage` requires:
- `env_config`: Dict passed to `LunarLanderEnv.__init__()` (NOT arbitrary parameters)
- `success_threshold`: Mean reward over `min_episodes` to advance (negative values OK for early stages)
- `max_timesteps`: Hard cap per stage (prevents infinite loops)

**Convention**: Stages progressively increase `initial_altitude_range`, `num_craters`, and reduce `initial_velocity_range` tolerance

### Basilisk Integration Quirks
1. **Path setup required**: All scripts must add `basilisk/dist3` to `sys.path` before importing
2. **Time units**: Basilisk uses nanoseconds (`macros.sec2nano(0.1)` for 0.1s timestep)
3. **Coordinate frames**: 
   - `_N`: Moon-centered inertial frame (North-East-Down)
   - `_B`: Body-fixed frame (+Z is nose/up, +X forward)
   - **Critical**: LIDAR scans in body frame, positions in inertial frame
4. **Fuel depletion**: `FuelTank` effectors auto-update spacecraft mass/inertia - DO NOT manually modify `lander.hub.mHub`

### Action Space Mapping (Compact Mode)
**Located in**: `lunar_lander_env.py::step()`
```python
# action = [main_throttle, pitch_torque, yaw_torque, roll_torque]
primary_throttles = [main_throttle] * 3  # Apply to all 3 engines
# Torques converted to differential RCS throttles (see step() for logic)
```
**Why compact**: 4D action space trains 10x faster than 9D full mode; sufficient for landing task

### File Output Structure (Auto-generated)
```
models/
├── best_model/           # Best model via EvalCallback (auto-saved)
├── checkpoints/          # Every 50k steps (ppo_lunar_lander_NNNNNN_steps.zip)
├── curriculum_final.zip  # Final curriculum model
└── stage*_final.zip      # Per-stage completions

logs/
└── PPO_1/                # TensorBoard logs (timestamped subdirs)
```

## Common Pitfalls & Solutions

### "Module not found: Basilisk"
**Cause**: Missing `sys.path` setup
**Fix**: Ensure all scripts include:
```python
import os, sys
basiliskPath = os.path.join(os.path.dirname(__file__), 'basilisk', 'dist3')
sys.path.insert(0, basiliskPath)
```

### Agent Not Learning (Reward < -200)
1. **Check curriculum advancement**: Use `--mode curriculum` (NOT `--mode standard` initially)
2. **Verify environment termination**: Ensure episodes end (check `max_episode_steps` in `LunarLanderEnv`)
3. **Inspect TensorBoard**: Look for `rollout/ep_rew_mean` trend over 100k+ steps
4. **Reduce action space**: Use `action_mode='compact'` (4D) not `'full'` (9D)

### Training Crashes on Step
**Likely**: Basilisk simulation divergence due to extreme actions
**Debug**: Add reward clipping in `_compute_reward()` or increase action smoothing (not currently implemented)

### Slow Training (< 1000 steps/sec)
- **Reduce parallel envs**: `--n-envs 2` on low-memory systems
- **Use SAC/TD3**: More sample-efficient than PPO (`--algorithm sac`)
- **Disable logging**: Comment out verbose TensorBoard logging in callbacks

## Key Files for Common Tasks

### Modifying Reward Function
Edit: `lunar_lander_env.py::_compute_reward()`
Test: `python lunar_lander_env.py` (runs 5-step sanity check)

### Adding Curriculum Stages
Edit: `unified_training.py::_create_curriculum()`
Test: `python unified_training.py --mode curriculum --start-stage N` (skip to stage N)

### Changing Terrain Difficulty
1. Generate new terrain: `python generate_terrain.py --craters 50 --size 5000`
2. Update `terrain_config` in curriculum stages or env init

### Hyperparameter Tuning
Edit: `unified_training.py::_create_model()` (learning_rate, n_steps, batch_size, etc.)
Reference: SB3 docs for algorithm-specific params

## Testing Philosophy
- **No unit tests** (physics simulation hard to mock)
- **Integration testing**: `--mode test` runs 1 episode with random actions
- **Validation**: Curriculum auto-advances only when reward threshold met across min_episodes
- **Visual inspection**: Use `--mode eval --render` to watch agent behavior

## Performance Expectations
- **Curriculum training**: 4-8 hours on modern CPU (16 cores, `--n-envs 8`)
- **Final success rate**: 70-85% successful landings on extreme terrain
- **Mean reward**: 600+ (includes landing bonus + efficiency bonuses)
- **Fuel efficiency**: 40-50% remaining at landing

## When Asking Questions
1. **For training issues**: Include TensorBoard metrics (`rollout/ep_rew_mean` trend) and command used
2. **For environment errors**: Run `python lunar_lander_env.py` and paste full traceback
3. **For reward tuning**: Specify target behavior (e.g., "prioritize fuel over precision")
4. **For curriculum changes**: State initial/target altitude, terrain difficulty, expected stage duration
